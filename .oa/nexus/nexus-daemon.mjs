#!/usr/bin/env node
/**
 * nexus-daemon.mjs — Standalone nexus process with real TCP/UDP sockets.
 * Spawned by the open-agents nexus tool. Communicates via JSON files.
 * v1.5.0: capability registration, peer blocking, metering, room members.
 *
 * Usage: node nexus-daemon.mjs <nexus-dir> <agent-name> [agent-type]
 */

import { NexusClient } from 'open-agents-nexus';
import { readFileSync, writeFileSync, existsSync, mkdirSync, unlinkSync, readdirSync, appendFileSync, watch as fsWatch } from 'node:fs';
import { join } from 'node:path';
import { homedir, hostname } from 'node:os';
import { createHash, createHmac } from 'node:crypto';

const nexusDir = process.argv[2];
const agentName = process.argv[3] || ('oa-node-' + process.pid);
const agentType = process.argv[4] || 'general';
const cmdFile = join(nexusDir, 'cmd.json');
const respFile = join(nexusDir, 'resp.json');
const statusFile = join(nexusDir, 'status.json');
const inboxDir = join(nexusDir, 'inbox');
const logFile = join(nexusDir, 'daemon.log');
function dlog(msg) { try { appendFileSync(logFile, new Date().toISOString() + ' ' + msg + '\n'); } catch {} }
const pidFile = join(nexusDir, 'daemon.pid');
const invocationsDir = join(nexusDir, 'invocations');
const meteringFile = join(nexusDir, 'metering.jsonl');
const x402KeyPath = join(nexusDir, 'x402-wallet.key');
const hasX402Key = existsSync(x402KeyPath);

mkdirSync(inboxDir, { recursive: true });
mkdirSync(invocationsDir, { recursive: true });

// Write PID so the agent can kill us
writeFileSync(pidFile, String(process.pid));

// Use GLOBAL identity key so all OA instances on this machine share one peerId.
// Fallback to project-scoped key if global doesn't exist.
const globalKeyDir = join(homedir(), '.open-agents');
const globalKeyPath = join(globalKeyDir, 'identity.key');
const projectKeyPath = join(nexusDir, 'identity.key');
const keyPath = existsSync(globalKeyPath) ? globalKeyPath : projectKeyPath;
var nexusOpts = {
  keyStorePath: keyPath,
  agentName,
  agentType,
  role: 'light',
  enableMdns: true,
  enablePubsubDiscovery: true,
  enableNats: true,
  natsServers: ['wss://demo.nats.io:8443'],
  enableCircuitRelay: true,
  usePublicBootstrap: true,
  trustPolicy: { denylist: [], allowlist: [] },
};
if (hasX402Key) {
  nexusOpts.x402 = {
    enabled: true,
    walletKeyPath: x402KeyPath,
    maxPaymentPerRequest: '5000000',
  };
  if (process.env.ALCHEMY_API_KEY) {
    nexusOpts.x402.alchemyApiKey = process.env.ALCHEMY_API_KEY;
  }
}
const nexus = new NexusClient(nexusOpts);

const rooms = new Map();
let connected = false;
const blockedPeers = [];
let cohereActive = false;
const cohereDedup = new Map(); // queryId -> timestamp (60s TTL dedup)
const cohereClaims = new Map(); // queryId -> { claimedBy: peerId, timestamp } (WO-1.3 claim protocol)
var _cLastModel = ''; // warm model tracking — last model used for any inference

// ── COHERE stats tracking ────────────────────────────────────────────
var _cohereStats = {
  queriesReceived: 0,
  queriesAnswered: 0,
  queriesErrors: 0,
  queriesSent: 0,
  totalLatencyMs: 0,
  modelsUsed: {},      // model -> count
  peersServed: {},     // peerId -> count
  bytesIn: 0,
  bytesOut: 0,
  startedAt: Date.now(),
  lastQueryAt: 0,
};
var _cohereStatsFile = join(nexusDir, 'cohere-stats.json');
function _saveStats() { try { writeFileSync(_cohereStatsFile, JSON.stringify(_cohereStats, null, 2)); } catch {} }
// Load persisted stats on startup
try { if (existsSync(_cohereStatsFile)) Object.assign(_cohereStats, JSON.parse(readFileSync(_cohereStatsFile, 'utf8'))); } catch {}

// ── COHERE model allowlist ───────────────────────────────────────────
// null = all models allowed, Set = only these models served to remote queries
var _cohereAllowedModels = null; // null means all
var _cohereModelsFile = join(nexusDir, 'cohere-models.json');
try {
  if (existsSync(_cohereModelsFile)) {
    var _cmData = JSON.parse(readFileSync(_cohereModelsFile, 'utf8'));
    if (Array.isArray(_cmData) && _cmData.length > 0) _cohereAllowedModels = new Set(_cmData);
  }
} catch {}
function _saveModelAllowlist() {
  try { writeFileSync(_cohereModelsFile, JSON.stringify(_cohereAllowedModels ? [..._cohereAllowedModels] : [], null, 2)); } catch {}
}

// ── IPFS/Helia (lazy init) ───────────────────────────────────────────
var _heliaNode = null;
var _heliaFs = null;
var _heliaReady = false;
var _heliaInitPromise = null;
var _ipfsDataDir = join(nexusDir, 'ipfs');
async function _ensureHelia() {
  if (_heliaReady) return true;
  if (_heliaInitPromise) return _heliaInitPromise;
  _heliaInitPromise = (async () => {
    try {
      mkdirSync(_ipfsDataDir, { recursive: true });
      var { createHelia } = await import('helia');
      var { unixfs } = await import('@helia/unixfs');
      var { FsBlockstore } = await import('blockstore-fs');
      var blockstore = new FsBlockstore(join(_ipfsDataDir, 'blocks'));
      _heliaNode = await createHelia({ blockstore });
      _heliaFs = unixfs(_heliaNode);
      _heliaReady = true;
      dlog('IPFS/Helia initialized — datastore: ' + _ipfsDataDir);
      return true;
    } catch (err) {
      dlog('IPFS/Helia init failed (graceful degradation): ' + (err.message || err));
      _heliaReady = false;
      _heliaInitPromise = null;
      return false;
    }
  })();
  return _heliaInitPromise;
}

// NATS invoke relay — for cross-NAT fallback when direct libp2p dial fails.
// Both peers connect to wss://demo.nats.io:8443, so NATS request/reply bridges the gap.
var _natsConn = null;
var _natsCodec = null;
var _tokensByRequest = {};

// Check if an error is a connectivity/dial failure (should trigger NATS fallback)
// vs an application-level error (auth rejected, capability not found — no point retrying via NATS)
function isDialFailure(errMsg) {
  var appErrors = ['unauthorized', 'capability not found', 'not registered', 'auth rejected'];
  var lower = errMsg.toLowerCase();
  for (var i = 0; i < appErrors.length; i++) {
    if (lower.includes(appErrors[i])) return false;
  }
  // Any network/transport error should fall back to NATS
  return lower.includes('multiaddrs failed') || lower.includes('econnrefused') ||
    lower.includes('etimedout') || lower.includes('dial to peer') ||
    lower.includes('no route') || lower.includes('circuit relay') ||
    lower.includes('stream was reset') || lower.includes('timeout') ||
    lower.includes('connection reset') || lower.includes('ehostunreach') ||
    lower.includes('enetunreach') || lower.includes('peer not reachable') ||
    lower.includes('dial error') || lower.includes('aborted') ||
    lower.includes('connection refused');
}

function writeStatus(extra = {}) {
  const caps = typeof nexus.getRegisteredCapabilities === 'function'
    ? nexus.getRegisteredCapabilities() : [];
  const data = {
    pid: process.pid,
    connected,
    peerId: connected ? nexus.peerId : null,
    agentName,
    agentType,
    rooms: [...rooms.keys()],
    capabilities: caps,
    blockedPeers,
    connectedAt: connected ? new Date().toISOString() : null,
    ...extra,
  };
  try { writeFileSync(statusFile, JSON.stringify(data, null, 2)); } catch {}
}

function writeResp(id, result) {
  try { writeFileSync(respFile, JSON.stringify({ id, ...result }, null, 2)); } catch {}
}

// Collect CPU/GPU/memory metrics for piggybacking on inference responses.
// Avoids separate invoke_capability round-trip that clogs the IPC.
var _sysMetricsCache = null;
var _sysMetricsCacheTs = 0;
async function _collectSysMetrics() {
  // Cache for 5s — nvidia-smi is expensive, inference can fire rapidly
  var now = Date.now();
  if (_sysMetricsCache && (now - _sysMetricsCacheTs) < 5000) return _sysMetricsCache;
  try {
    var os = await import('node:os');
    var loads = os.loadavg();
    var cores = os.cpus().length;
    var totalMem = os.totalmem();
    var freeMem = os.freemem();
    var usedMem = totalMem - freeMem;
    var cpuModel = '';
    try { var cpuArr = os.cpus(); if (cpuArr.length > 0) cpuModel = cpuArr[0].model || ''; } catch {}
    var gpuInfo = { available: false, name: '', utilization: 0, vramUsedMB: 0, vramTotalMB: 0, vramUtilization: 0 };
    try {
      var cp = await import('node:child_process');
      var smiOut = cp.execSync('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,name --format=csv,noheader,nounits 2>/dev/null', { encoding: 'utf8', timeout: 3000 });
      var smiLine = smiOut.trim().split('\n')[0];
      if (smiLine) {
        var sp = smiLine.split(',').map(function(s) { return s.trim(); });
        gpuInfo.available = true;
        gpuInfo.utilization = parseInt(sp[0] || '0', 10) || 0;
        gpuInfo.vramUsedMB = parseInt(sp[1] || '0', 10) || 0;
        gpuInfo.vramTotalMB = parseInt(sp[2] || '0', 10) || 0;
        gpuInfo.name = sp[3] || '';
        gpuInfo.vramUtilization = gpuInfo.vramTotalMB > 0 ? Math.round((gpuInfo.vramUsedMB / gpuInfo.vramTotalMB) * 100) : 0;
      }
    } catch {}
    _sysMetricsCache = {
      cpu: { utilization: Math.min(100, Math.round((loads[0] / cores) * 100)), cores: cores, model: cpuModel },
      memory: { utilization: Math.round((usedMem / totalMem) * 100), totalGB: Math.round((totalMem / (1024*1024*1024)) * 10) / 10, usedGB: Math.round((usedMem / (1024*1024*1024)) * 10) / 10 },
      gpu: gpuInfo,
      timestamp: new Date().toISOString(),
    };
    _sysMetricsCacheTs = now;
    return _sysMetricsCache;
  } catch { return null; }
}

async function handleCmd(cmd) {
  const { id, action, args } = cmd;
  dlog('handleCmd: action=' + action + ' id=' + id);
  try {
    switch (action) {
      case 'join_room': {
        const roomId = args.room_id;
        if (rooms.has(roomId)) { writeResp(id, { ok: true, output: 'Already in room: ' + roomId }); return; }
        const room = await nexus.joinRoom(roomId);
        rooms.set(roomId, room);
        // Per-room message listener (also captured by client-level event)
        room.on('message', (msg) => {
          const roomInbox = join(inboxDir, roomId);
          mkdirSync(roomInbox, { recursive: true });
          const fname = Date.now() + '-' + (msg.id || '').slice(0, 8) + '.json';
          const entry = {
            sender: msg.sender,
            content: msg.payload?.content || '',
            format: msg.payload?.format || 'text/plain',
            timestamp: msg.timestamp || Date.now(),
            id: msg.id,
          };
          try { writeFileSync(join(roomInbox, fname), JSON.stringify(entry, null, 2)); } catch {}
        });
        // v1.5.0: Room member tracking
        if (room.on) {
          room.on('member:join', (member) => {
            console.log('Member joined ' + roomId + ': ' + (member.agentName || member.peerId));
          });
          room.on('member:leave', (member) => {
            console.log('Member left ' + roomId + ': ' + (member.agentName || member.peerId));
          });
        }
        writeStatus();
        writeResp(id, { ok: true, output: 'Joined room: ' + roomId });
        break;
      }
      case 'leave_room': {
        const roomId = args.room_id;
        const room = rooms.get(roomId);
        if (!room) { writeResp(id, { ok: false, output: 'Not in room: ' + roomId }); return; }
        await room.leave();
        rooms.delete(roomId);
        writeStatus();
        writeResp(id, { ok: true, output: 'Left room: ' + roomId });
        break;
      }
      case 'send_message': {
        const room = rooms.get(args.room_id);
        if (!room) { writeResp(id, { ok: false, output: 'Not in room: ' + args.room_id + '. Join it first.' }); return; }
        const msgId = await room.send(args.message, { format: 'text/plain' });
        // Relay to NATS for frontend visibility (public room messages)
        if (_natsConn && _natsCodec) {
          try {
            _natsConn.publish('nexus.rooms.chat', _natsCodec.encode(JSON.stringify({
              type: 'nexus.room.message',
              roomId: args.room_id,
              peerId: nexus.peerId,
              agentName: agentName,
              content: String(args.message).slice(0, 500),
              timestamp: Date.now(),
            })));
          } catch (natsRelayErr) {
            dlog('NATS chat relay error: ' + (natsRelayErr.message || natsRelayErr));
          }
        }
        writeResp(id, { ok: true, output: 'Message sent (id: ' + msgId + ')' });
        break;
      }
      // ── Sponsor announcement via NATS ──
      case 'sponsor_announce': {
        // Publish sponsor metadata to NATS so other OA instances can discover it
        if (!_natsConn || !_natsCodec) {
          writeResp(id, { ok: false, output: 'NATS not connected — cannot announce sponsorship' });
          return;
        }
        var sponsorData = {
          type: 'sponsor.announce',
          peerId: globalThis._daemonPeerId || 'unknown',
          libp2pPeerId: globalThis._daemonPeerId || '',
          name: args.name || 'Anonymous Sponsor',
          models: args.models || [],
          tunnelUrl: args.tunnel_url || null,
          authKey: args.auth_key || '',
          limits: {
            maxRequestsPerMinute: parseInt(args.rpm || '60', 10),
            maxTokensPerDay: parseInt(args.tpd || '100000', 10),
          },
          banner: args.banner || null,
          message: args.message || '',
          linkUrl: args.link_url || '',
          linkText: args.link_text || '',
          status: 'active',
          timestamp: Date.now(),
        };
        _natsConn.publish('nexus.sponsors.announce', _natsCodec.encode(JSON.stringify(sponsorData)));
        // Store for periodic re-announcement (every capacity tick, ~60s)
        globalThis._activeSponsorData = sponsorData;
        dlog('sponsor_announce: published to nexus.sponsors.announce');

        // Persist to KV-backed sponsor directory (openagents.nexus worker)
        try {
          var kvResp = await fetch('https://openagents.nexus/api/v1/sponsors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sponsorData),
          });
          var kvResult = await kvResp.json();
          dlog('sponsor_announce: KV persist ' + (kvResult.persisted ? 'OK' : 'skipped: ' + kvResult.reason));
        } catch (kvErr) {
          dlog('sponsor_announce: KV persist failed: ' + (kvErr.message || kvErr));
        }
        // Also join the sponsors room and send as room message for persistence
        if (!rooms.has('sponsors')) {
          try {
            var spRoom = await nexus.joinRoom('sponsors');
            rooms.set('sponsors', spRoom);
            dlog('sponsor_announce: auto-joined sponsors room');
          } catch (roomErr) {
            dlog('sponsor_announce: room join failed: ' + (roomErr.message || roomErr));
          }
        }
        if (rooms.has('sponsors')) {
          try {
            await rooms.get('sponsors').send(JSON.stringify(sponsorData), { format: 'application/json' });
          } catch {}
        }
        writeResp(id, { ok: true, output: 'Sponsor announced: ' + sponsorData.name + ' (' + sponsorData.models.length + ' models)' });
        break;
      }
      case 'sponsor_discover': {
        var foundSponsors = [];
        var discoverTimeout = parseInt(args.timeout_ms || '5000', 10);

        // ── Source 1: KV-backed persistent directory (MOST RELIABLE) ──
        // Query the openagents.nexus worker for persisted sponsor listings
        try {
          var kvResp = await fetch('https://openagents.nexus/api/v1/sponsors', { signal: AbortSignal.timeout(5000) });
          if (kvResp.ok) {
            var kvData = await kvResp.json();
            var kvSponsors = kvData.sponsors || [];
            for (var ki = 0; ki < kvSponsors.length; ki++) {
              var kvSp = kvSponsors[ki];
              if (kvSp.status === 'active') {
                kvSp.type = 'sponsor.announce'; // normalize
                foundSponsors.push(kvSp);
              }
            }
            dlog('sponsor_discover: KV directory returned ' + kvSponsors.length + ' sponsor(s)');
          }
        } catch (kvErr) {
          dlog('sponsor_discover: KV fetch failed: ' + (kvErr.message || kvErr));
        }

        // ── Source 2: NATS live announcements (if connected) ──
        if (_natsConn && _natsCodec) {

        // Also check sponsors room inbox for cached messages
        var sponsorInbox = join(inboxDir, 'sponsors');
        try {
          if (existsSync(sponsorInbox)) {
            var inboxFiles = readdirSync(sponsorInbox).filter(f => f.endsWith('.json')).sort().reverse().slice(0, 20);
            for (var fi = 0; fi < inboxFiles.length; fi++) {
              try {
                var msg = JSON.parse(readFileSync(join(sponsorInbox, inboxFiles[fi]), 'utf8'));
                var content = msg.content || '';
                try {
                  var parsed = JSON.parse(content);
                  if (parsed.type === 'sponsor.announce' && parsed.status === 'active') {
                    // Only include if not too stale (< 10 minutes)
                    if (Date.now() - (parsed.timestamp || 0) < 600000) {
                      foundSponsors.push(parsed);
                    }
                  }
                } catch {}
              } catch {}
            }
          }
        } catch {}

        // Join sponsors room to receive future announcements
        if (!rooms.has('sponsors')) {
          try {
            var spRoom2 = await nexus.joinRoom('sponsors');
            rooms.set('sponsors', spRoom2);
          } catch {}
        }

        // Subscribe to NATS for live announcements (short window)
        try {
          var sub = _natsConn.subscribe('nexus.sponsors.announce');
          var subDone = false;
          setTimeout(() => { subDone = true; sub.unsubscribe(); }, discoverTimeout);
          for await (var natMsg of sub) {
            if (subDone) break;
            try {
              var sp = JSON.parse(_natsCodec.decode(natMsg.data));
              if (sp.type === 'sponsor.announce' && sp.status === 'active') {
                foundSponsors.push(sp);
              }
            } catch {}
          }
        } catch (subErr) {
          dlog('sponsor_discover: NATS sub error: ' + (subErr.message || subErr));
        }

        } // end if (_natsConn && _natsCodec)

        // Deduplicate by peerId
        var seen = {};
        var unique = [];
        for (var si = 0; si < foundSponsors.length; si++) {
          var key = foundSponsors[si].peerId || foundSponsors[si].tunnelUrl || ('sp-' + si);
          if (!seen[key]) {
            seen[key] = true;
            unique.push(foundSponsors[si]);
          }
        }

        writeResp(id, { ok: true, output: unique.length + ' sponsor(s) found', sponsors: unique });
        break;
      }
      case 'discover_peers': {
        const node = nexus.network?.node;
        const peers = node?.getPeers?.() || [];
        const list = peers.slice(0, 20).map(p => p.toString?.() || String(p));
        // Include NATS-discovered peers with capabilities
        var dpNatsFile = join(nexusDir, 'discovered-peers.json');
        var dpNats = {};
        try { if (existsSync(dpNatsFile)) dpNats = JSON.parse(readFileSync(dpNatsFile, 'utf8')); } catch {}
        var dpNatsCount = Object.keys(dpNats).length;
        var dpLines = ['Connected peers: ' + list.length];
        for (var dpi = 0; dpi < list.length; dpi++) {
          dpLines.push('  ' + list[dpi]);
        }
        if (dpNatsCount > 0) {
          dpLines.push('NATS-discovered peers: ' + dpNatsCount);
          for (var dpKey of Object.keys(dpNats)) {
            var dpEntry = dpNats[dpKey];
            var dpAge = Math.round((Date.now() - dpEntry.lastSeen) / 1000);
            dpLines.push('  ' + dpKey.slice(0, 20) + '... (' + dpEntry.agentName + ') caps=' + (dpEntry.capabilities || []).length + ' seen=' + dpAge + 's ago');
          }
        }
        writeResp(id, { ok: true, output: dpLines.join('\n') });
        break;
      }
      case 'discover_peer_caps': {
        var dpcPeerId = args.peer_id;
        if (!dpcPeerId) { writeResp(id, { ok: false, output: 'peer_id is required' }); return; }
        var dpcFile = join(nexusDir, 'discovered-peers.json');
        var dpcPeers = {};
        try { if (existsSync(dpcFile)) dpcPeers = JSON.parse(readFileSync(dpcFile, 'utf8')); } catch {}
        var dpcEntry = dpcPeers[String(dpcPeerId)];
        if (dpcEntry && dpcEntry.capabilities && dpcEntry.capabilities.length > 0) {
          var dpcAge = Date.now() - dpcEntry.lastSeen;
          if (dpcAge < 5 * 60 * 1000) { // 5 minute freshness
            writeResp(id, { ok: true, output: JSON.stringify(dpcEntry, null, 2) });
          } else {
            writeResp(id, { ok: false, output: 'Peer data stale (' + Math.round(dpcAge / 1000) + 's old). Wait for NATS re-announce.' });
          }
        } else {
          writeResp(id, { ok: false, output: 'Peer not found in NATS discovery cache. Peer may not have announced yet (announcements every 30s).' });
        }
        break;
      }
      case 'list_rooms': {
        const joined = [...rooms.keys()];
        writeResp(id, { ok: true, output: joined.length ? 'Rooms: ' + joined.join(', ') : 'No rooms joined.' });
        break;
      }
      case 'send_dm': {
        const peerId = args.target_peer;
        if (!peerId) { writeResp(id, { ok: false, output: 'target_peer is required' }); return; }
        await nexus.sendDM(peerId, args.message || '');
        writeResp(id, { ok: true, output: 'DM sent to ' + peerId.slice(0, 20) + '...' });
        break;
      }
      case 'find_agent': {
        const peerId = args.peer_id;
        if (!peerId) { writeResp(id, { ok: false, output: 'peer_id is required' }); return; }
        try {
          var FIND_TIMEOUT = 15000;
          var findPromise = nexus.findAgent(peerId);
          var findTimeoutPromise = new Promise(function(resolve) { setTimeout(function() { resolve(null); }, FIND_TIMEOUT); });
          var profile = await Promise.race([findPromise, findTimeoutPromise]);
          writeResp(id, { ok: true, output: profile ? JSON.stringify(profile, null, 2) : 'Agent not found: ' + peerId });
        } catch (findErr) {
          writeResp(id, { ok: false, output: 'find_agent error: ' + (findErr.message || String(findErr)) });
        }
        break;
      }
      case 'query_peer_caps': {
        var qpcPeerId = args.peer_id;
        if (!qpcPeerId) { writeResp(id, { ok: false, output: 'peer_id is required' }); return; }
        dlog('query_peer_caps: peer=' + String(qpcPeerId).slice(0, 20));
        try {
          var QPC_TIMEOUT = 15000;
          var qpcInput = {};
          if (args.auth_key) qpcInput.auth_key = args.auth_key;

          // Race libp2p + NATS in parallel — fastest route wins
          var qpcLibp2p = nexus.invokeCapability(String(qpcPeerId), '__list_capabilities', qpcInput, { stream: false, maxDurationMs: 10000 });
          var qpcTimeout = new Promise(function(_, reject) { setTimeout(function() { reject(new Error('query_peer_caps timed out')); }, QPC_TIMEOUT); });
          var qpcRace = [qpcLibp2p, qpcTimeout];

          if (_natsConn && _natsCodec) {
            var _qpcNatsP = (async function() {
              var _s = 'nexus.invoke.' + String(qpcPeerId);
              var _p = JSON.stringify({ capability: '__list_capabilities', input: qpcInput, from: nexus.peerId || '' });
              var _r = await _natsConn.request(_s, _natsCodec.encode(_p), { timeout: 15000 });
              var _res = JSON.parse(_natsCodec.decode(_r.data));
              if (_res && _res.error) throw new Error('NATS: ' + _res.error);
              dlog('query_peer_caps: NATS won the race');
              return _res;
            })();
            qpcRace.push(_qpcNatsP);
          }

          var qpcResult = await Promise.any(qpcRace.map(function(p) {
            return p.catch(function(e) { return Promise.reject(e); });
          }));
          dlog('query_peer_caps: SUCCESS');
          writeResp(id, { ok: true, output: typeof qpcResult === 'string' ? qpcResult : JSON.stringify(qpcResult, null, 2) });
        } catch (qpcErr) {
          var qpcErrMsg = qpcErr.errors ? qpcErr.errors.map(function(e) { return e.message || String(e); }).join('; ') : (qpcErr.message || String(qpcErr));
          dlog('query_peer_caps: ALL paths failed: ' + qpcErrMsg);
          writeResp(id, { ok: false, output: 'query_peer_caps error: ' + qpcErrMsg });
        }
        break;
      }
      case 'invoke_capability': {
        const peerId = args.target_peer;
        const capability = args.capability || 'text-generation';
        const input = args.input || {};
        if (!peerId) { writeResp(id, { ok: false, output: 'target_peer is required' }); return; }
        dlog('invoke_capability: peer=' + peerId.slice(0, 20) + ' cap=' + capability);
        try {
          // Race libp2p + NATS in parallel — whichever route works first wins
          const INVOKE_TIMEOUT = 300_000;
          const invokeInput = typeof input === 'string' ? { prompt: input } : input;
          const invokePromise = nexus.invokeCapability(peerId, capability, invokeInput, { stream: false, maxDurationMs: 240000 });
          const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error('invoke_capability timed out after ' + (INVOKE_TIMEOUT / 1000) + 's')), INVOKE_TIMEOUT));

          var icRaceCandidates = [invokePromise, timeoutPromise];

          if (_natsConn && _natsCodec) {
            var _icNatsRaceP = (async function() {
              var _s = 'nexus.invoke.' + peerId;
              var _p = JSON.stringify({ capability: capability, input: invokeInput, from: nexus.peerId || '' });
              var _r = await _natsConn.request(_s, _natsCodec.encode(_p), { timeout: 240000 });
              var _res = JSON.parse(_natsCodec.decode(_r.data));
              if (_res && _res.error) throw new Error('NATS: ' + _res.error);
              dlog('invoke_capability: NATS won the race');
              return _res;
            })();
            icRaceCandidates.push(_icNatsRaceP);
          }

          const result = await Promise.any(icRaceCandidates.map(function(p) {
            return p.catch(function(e) { return Promise.reject(e); });
          }));
          dlog('invoke_capability: SUCCESS');
          writeResp(id, { ok: true, output: typeof result === 'string' ? result : JSON.stringify(result, null, 2) });
        } catch (invokeErr) {
          var invokeErrMsg = invokeErr.errors ? invokeErr.errors.map(function(e) { return e.message || String(e); }).join('; ') : (invokeErr.message || String(invokeErr));
          dlog('invoke_capability: ALL paths failed: ' + invokeErrMsg);
          writeResp(id, { ok: false, output: 'Invoke error: ' + invokeErrMsg });
        }
        break;
      }
      case 'remote_infer': {
        var riModel = args.model;
        var riPrompt = args.prompt || args.input || '';
        var riMessages = args.messages || null;
        var riTools = args.tools || null;
        var riTargetPeer = args.target_peer || '';
        if (!riModel) { writeResp(id, { ok: false, output: 'model is required' }); return; }
        if (!riPrompt && !riMessages) { writeResp(id, { ok: false, output: 'prompt or messages is required' }); return; }

        // Build invoke data — structured (messages+tools) or flat (prompt)
        var riData;
        if (riMessages) {
          var riParsedMsgs = typeof riMessages === 'string' ? JSON.parse(riMessages) : riMessages;
          riData = { messages: riParsedMsgs };
          if (riTools) {
            riData.tools = typeof riTools === 'string' ? JSON.parse(riTools) : riTools;
          }
          if (args.temperature !== undefined) riData.temperature = Number(args.temperature);
          if (args.max_tokens) riData.max_tokens = Number(args.max_tokens);
          if (args.think !== undefined) riData.think = args.think === 'true' || args.think === true;
        } else {
          riData = { prompt: riPrompt };
        }

        var riAuthKey = args.auth_key || '';

        // Derive capability name from model
        var riCapName = 'inference:' + riModel.replace(/[^a-zA-Z0-9._-]/g, '_');
        dlog('remote_infer: model=' + riModel + ' cap=' + riCapName + ' prompt_len=' + riPrompt.length + (riAuthKey ? ' auth=yes' : ''));

        // Include auth key in data if provided
        if (riAuthKey) {
          riData.auth_key = riAuthKey;
        }

        // Streaming mode: if stream_file is provided, use stream:true and write tokens to file
        var riStreamFile = args.stream_file || '';
        var riUseStream = !!riStreamFile;

        // If target_peer specified, invoke directly. Otherwise auto-discover.
        if (riTargetPeer) {
          dlog('remote_infer: invoking ' + riCapName + ' on peer ' + riTargetPeer.slice(0, 20) + ' stream=' + riUseStream + ' (libp2p' + (_natsConn ? ' + NATS parallel' : '') + ')');
          try {
            var RI_TIMEOUT = 120000;
            var riInvokeP = nexus.invokeCapability(riTargetPeer, riCapName, riData, { stream: riUseStream, maxDurationMs: 90000 });
            var riTimeoutP = new Promise(function(_, reject) { setTimeout(function() { reject(new Error('remote_infer timed out after ' + (RI_TIMEOUT / 1000) + 's')); }, RI_TIMEOUT); });

            // Build race candidates — libp2p + timeout always present
            var riRaceCandidates = [riInvokeP, riTimeoutP];

            // Add NATS as a parallel candidate if available
            if (_natsConn && _natsCodec) {
              var _riNatsRaceP = (async function() {
                var _rSubject = 'nexus.invoke.' + riTargetPeer;
                var _rPayload = JSON.stringify({ capability: riCapName, input: riData, from: nexus.peerId || '' });
                var _rResp = await _natsConn.request(_rSubject, _natsCodec.encode(_rPayload), { timeout: 45000 });
                var _rResult = JSON.parse(_natsCodec.decode(_rResp.data));
                if (_rResult && _rResult.error) throw new Error('NATS: ' + _rResult.error);
                dlog('remote_infer: NATS won the race');
                return _rResult;
              })();
              riRaceCandidates.push(_riNatsRaceP);
            }

            var riResult = await Promise.any(riRaceCandidates.map(function(p) {
              return p.catch(function(e) { return Promise.reject(e); });
            }));
            dlog('remote_infer: SUCCESS (peer ' + riTargetPeer.slice(0, 20) + ')');
          } catch (riErr) {
            // Promise.any throws AggregateError when ALL candidates fail
            var riErrMsg = riErr.errors ? riErr.errors.map(function(e) { return e.message || String(e); }).join('; ') : (riErr.message || String(riErr));
            dlog('remote_infer: ALL paths failed: ' + riErrMsg);
            writeResp(id, { ok: false, output: 'Remote inference failed: ' + riErrMsg });
            return;
          }
        } else {
          // Auto-discover: try speculative invoke on each 12D3KooW peer
          // Skip Qm-prefixed bootstrap/relay nodes — they never host inference
          var discStart = Date.now();
          var riResult = null;
          try {
            var riNode = nexus.network ? nexus.network.node : null;
            var riPeers = riNode && typeof riNode.getPeers === 'function' ? riNode.getPeers() : [];
            var riCandidates = [];
            for (var rfi = 0; rfi < riPeers.length; rfi++) {
              var rpStr = riPeers[rfi].toString ? riPeers[rfi].toString() : String(riPeers[rfi]);
              if (rpStr.startsWith('12D3KooW')) riCandidates.push(rpStr);
            }
            dlog('remote_infer: auto-discover across ' + riCandidates.length + ' peers (of ' + riPeers.length + ' total) for ' + riCapName);

            for (var ri = 0; ri < riCandidates.length && !riResult; ri++) {
              var rPid = riCandidates[ri];
              dlog('remote_infer: trying peer ' + rPid.slice(0, 20) + '...');
              try {
                var PEER_INVOKE_TIMEOUT = 30000;  // 30s per peer in discovery — fail fast, try next
                var specInvoke = nexus.invokeCapability(rPid, riCapName, riData, { stream: false, maxDurationMs: 25000 });
                var specTimeout = new Promise(function(_, reject) {
                  setTimeout(function() { reject(new Error('peer invoke timeout')); }, PEER_INVOKE_TIMEOUT);
                });
                riResult = await Promise.race([specInvoke, specTimeout]);
                riTargetPeer = rPid;
                dlog('remote_infer: SUCCESS via peer ' + rPid.slice(0, 20) + ' (' + (Date.now() - discStart) + 'ms)');
              } catch (specErr) {
                dlog('remote_infer: peer ' + rPid.slice(0, 20) + ' failed: ' + (specErr.message || specErr));
              }
            }
          } catch (discErr) {
            dlog('remote_infer: discovery error: ' + (discErr.message || discErr));
          }
          dlog('remote_infer: auto-discover took ' + (Date.now() - discStart) + 'ms');

          // If libp2p auto-discover failed, try NATS relay against discovered peers from cache
          if (!riResult && _natsConn && _natsCodec) {
            try {
              var _dpFile = join(nexusDir, 'discovered-peers.json');
              if (existsSync(_dpFile)) {
                var _dpData = JSON.parse(readFileSync(_dpFile, 'utf8'));
                var _dpPeers = Object.keys(_dpData).filter(function(p) { return p.startsWith('12D3KooW'); });
                dlog('remote_infer: trying NATS relay on ' + _dpPeers.length + ' cached peers');
                for (var _dpi = 0; _dpi < _dpPeers.length && !riResult; _dpi++) {
                  var _dpPid = _dpPeers[_dpi];
                  try {
                    var _dpSubject = 'nexus.invoke.' + _dpPid;
                    var _dpPayload = JSON.stringify({ capability: riCapName, input: riData, from: nexus.peerId || '' });
                    var _dpResp = await _natsConn.request(_dpSubject, _natsCodec.encode(_dpPayload), { timeout: 30000 });
                    var _dpResult = JSON.parse(_natsCodec.decode(_dpResp.data));
                    if (_dpResult && !_dpResult.error) {
                      riResult = _dpResult;
                      riTargetPeer = _dpPid;
                      dlog('remote_infer: SUCCESS via NATS relay peer ' + _dpPid.slice(0, 20));
                    }
                  } catch (_dpErr) {
                    dlog('remote_infer: NATS peer ' + _dpPid.slice(0, 20) + ' failed: ' + (_dpErr.message || _dpErr));
                  }
                }
              }
            } catch (_dpCacheErr) {
              dlog('remote_infer: NATS discovery cache error: ' + (_dpCacheErr.message || _dpCacheErr));
            }
          }

          if (!riResult) {
            writeResp(id, { ok: false, output: 'No peer found with capability: ' + riCapName + '. Ensure a provider is running expose with model ' + riModel });
            return;
          }
        }

        // riResult and riTargetPeer are available from either path above
        // ── Streaming mode: iterate AsyncGenerator, write tokens to file ──
        if (riUseStream && riResult && typeof riResult[Symbol.asyncIterator] === 'function') {
          // Signal OA process that streaming has started
          writeResp(id, { ok: true, output: JSON.stringify({ streaming: true, stream_file: riStreamFile }) });

          var riTokenContent = '';
          var riToolCalls = [];
          var riInputTokens = 0;
          var riOutputTokens = 0;
          try {
            for await (var riEvt of riResult) {
              if (riEvt.event === 'token' && riEvt.data) {
                var tkn = typeof riEvt.data === 'string' ? riEvt.data : String(riEvt.data);
                riTokenContent += tkn;
                appendFileSync(riStreamFile, JSON.stringify({ type: 'token', content: tkn }) + '\n');
              } else if (riEvt.event === 'result' && riEvt.data) {
                // Final metadata event — parse for usage/tool_calls/system
                try {
                  var riMeta = typeof riEvt.data === 'string' ? JSON.parse(riEvt.data) : riEvt.data;
                  if (riMeta.usage) {
                    riInputTokens = riMeta.usage.input_tokens || riMeta.usage.prompt_tokens || 0;
                    riOutputTokens = riMeta.usage.output_tokens || riMeta.usage.completion_tokens || 0;
                  }
                  if (riMeta.choices && riMeta.choices[0] && riMeta.choices[0].message && riMeta.choices[0].message.tool_calls) {
                    riToolCalls = riMeta.choices[0].message.tool_calls;
                  }
                  // Pass through provider system metrics
                  if (riMeta.system) var riSystemMetrics = riMeta.system;
                } catch {}
              }
            }
          } catch (riStreamErr) {
            dlog('remote_infer: stream error: ' + (riStreamErr.message || riStreamErr));
          }

          // Write done sentinel with final response
          var riFinalPayload = {
            model: riModel,
            response: riTokenContent,
            choices: [{
              message: {
                content: riTokenContent,
                tool_calls: riToolCalls.length > 0 ? riToolCalls : undefined,
              },
            }],
            usage: { input_tokens: riInputTokens, output_tokens: riOutputTokens },
            system: typeof riSystemMetrics !== 'undefined' ? riSystemMetrics : undefined,
          };
          appendFileSync(riStreamFile, JSON.stringify({ type: 'done', result: JSON.stringify(riFinalPayload) }) + '\n');
          // Write piggybacked system metrics to file for status bar
          if (typeof riSystemMetrics !== 'undefined' && riSystemMetrics) {
            try { writeFileSync(join(nexusDir, 'remote-metrics.json'), JSON.stringify({ ts: Date.now(), data: riSystemMetrics })); } catch {}
          }
          dlog('remote_infer: stream complete, tokens=' + riTokenContent.length + ' in=' + riInputTokens + ' out=' + riOutputTokens);
          break;
        }

        // ── Unary mode: check for errors and write response ──
        var riIsError = false;
        if (typeof riResult === 'string') {
          var riParsedCheck = null;
          try { riParsedCheck = JSON.parse(riResult); } catch {}
          if (riParsedCheck && typeof riParsedCheck === 'object' &&
              (riParsedCheck.model || riParsedCheck.choices || riParsedCheck.response !== undefined)) {
            riIsError = false;
          } else {
            var riLower = riResult.toLowerCase();
            if (riLower.includes('unauthorized') || riLower.includes('forbidden') ||
                riLower.includes('not found') || riLower.includes('error') ||
                riLower.includes('payment') || riLower.includes('rejected')) {
              riIsError = true;
            }
          }
        }
        if (riIsError) {
          writeResp(id, { ok: false, output: 'Remote peer error: ' + riResult });
        } else {
          var riOutput = {
            success: true,
            model: riModel,
            peer: riTargetPeer,
            capability: riCapName,
            result: riResult,
          };
          writeResp(id, { ok: true, output: JSON.stringify(riOutput, null, 2) });
          // Write piggybacked system metrics to file for status bar (avoids IPC invoke)
          try {
            var riParsedForSys = typeof riResult === 'string' ? JSON.parse(riResult) : riResult;
            if (riParsedForSys && riParsedForSys.system) {
              writeFileSync(join(nexusDir, 'remote-metrics.json'), JSON.stringify({ ts: Date.now(), data: riParsedForSys.system }));
            }
          } catch {}
        }
        break;
      }
      case 'store_content': {
        const data = args.data;
        if (!data) { writeResp(id, { ok: false, output: 'data is required' }); return; }
        const cid = await nexus.store(typeof data === 'string' ? { data } : data);
        writeResp(id, { ok: true, output: 'Stored. CID: ' + cid });
        break;
      }
      case 'retrieve_content': {
        const cid = args.cid;
        if (!cid) { writeResp(id, { ok: false, output: 'cid is required' }); return; }
        const data = await nexus.retrieve(cid);
        writeResp(id, { ok: true, output: JSON.stringify(data, null, 2) });
        break;
      }

      // ── v1.5.0: Capability Registration ──────────────────────────────
      case 'register_capability': {
        const name = args.capability;
        if (!name) { writeResp(id, { ok: false, output: 'capability name is required' }); return; }
        if (typeof nexus.registerCapability !== 'function') {
          writeResp(id, { ok: false, output: 'registerCapability not available (nexus version too old)' });
          return;
        }
        nexus.registerCapability(name, async (request, stream) => {
          // Log inbound invocation
          const logEntry = {
            ts: Date.now(),
            from: request.from || 'unknown',
            capability: request.capability || name,
            requestId: request.requestId,
          };
          const logFile = join(invocationsDir, Date.now() + '-' + name + '.json');
          try { writeFileSync(logFile, JSON.stringify(logEntry, null, 2)); } catch {}

          // Collect input chunks
          let inputData = '';
          stream.onData((msg) => {
            if (msg.type === 'invoke.chunk') {
              inputData += (typeof msg.data === 'string' ? msg.data : JSON.stringify(msg.data));
            }
          });

          // Accept the invocation
          await stream.write({
            type: 'invoke.accept', version: 1,
            requestId: request.requestId, accepted: true,
          });

          // Send event with acknowledgment
          await stream.write({
            type: 'invoke.event', version: 1,
            requestId: request.requestId, seq: 0,
            event: 'ack', data: agentName + ' received invocation for ' + name,
          });

          // Complete
          await stream.write({
            type: 'invoke.done', version: 1,
            requestId: request.requestId,
            usage: { inputBytes: inputData.length, outputBytes: 0 },
          });
          stream.close();
        });
        writeStatus();
        writeResp(id, { ok: true, output: 'Capability registered: ' + name + '. Now advertised via NATS.' });
        break;
      }
      case 'unregister_capability': {
        const name = args.capability;
        if (!name) { writeResp(id, { ok: false, output: 'capability name is required' }); return; }
        if (typeof nexus.unregisterCapability === 'function') {
          nexus.unregisterCapability(name);
        }
        writeStatus();
        writeResp(id, { ok: true, output: 'Capability unregistered: ' + name });
        break;
      }
      case 'list_capabilities': {
        const caps = typeof nexus.getRegisteredCapabilities === 'function'
          ? nexus.getRegisteredCapabilities() : [];
        writeResp(id, { ok: true, output: caps.length ? 'Capabilities: ' + caps.join(', ') : 'No capabilities registered.' });
        break;
      }

      // ── v1.5.0: Peer Blocking ────────────────────────────────────────
      case 'block_peer': {
        const peerId = args.target_peer || args.peer_id;
        if (!peerId) { writeResp(id, { ok: false, output: 'peer_id or target_peer is required' }); return; }
        if (typeof nexus.blockPeer === 'function') {
          nexus.blockPeer(peerId);
        }
        if (!blockedPeers.includes(peerId)) blockedPeers.push(peerId);
        writeStatus();
        writeResp(id, { ok: true, output: 'Blocked peer: ' + peerId.slice(0, 20) + '...' });
        break;
      }
      case 'unblock_peer': {
        const peerId = args.target_peer || args.peer_id;
        if (!peerId) { writeResp(id, { ok: false, output: 'peer_id or target_peer is required' }); return; }
        if (typeof nexus.unblockPeer === 'function') {
          nexus.unblockPeer(peerId);
        }
        const idx = blockedPeers.indexOf(peerId);
        if (idx >= 0) blockedPeers.splice(idx, 1);
        writeStatus();
        writeResp(id, { ok: true, output: 'Unblocked peer: ' + peerId.slice(0, 20) + '...' });
        break;
      }

      // ── v1.5.0: Metering ─────────────────────────────────────────────
      case 'metering_status': {
        if (!nexus.metering) {
          writeResp(id, { ok: true, output: 'Metering not available (nexus version too old).' });
          return;
        }
        const filter = {};
        if (args.peer_id) filter.peerId = args.peer_id;
        if (args.capability) filter.service = args.capability;

        // Try per-peer summary first
        if (args.peer_id && typeof nexus.metering.getSummary === 'function') {
          const summary = nexus.metering.getSummary(args.peer_id);
          if (summary) {
            writeResp(id, { ok: true, output: JSON.stringify(summary, null, 2) });
            return;
          }
        }

        // All summaries
        if (typeof nexus.metering.getAllSummaries === 'function') {
          const all = nexus.metering.getAllSummaries();
          if (all && (all.size > 0 || Object.keys(all).length > 0)) {
            const entries = all instanceof Map ? Object.fromEntries(all) : all;
            writeResp(id, { ok: true, output: 'Metering summaries:\n' + JSON.stringify(entries, null, 2) });
            return;
          }
        }

        // Fall back to raw records
        if (typeof nexus.metering.getRecords === 'function') {
          const records = nexus.metering.getRecords(filter);
          writeResp(id, { ok: true, output: 'Records: ' + records.length + '\n' + JSON.stringify(records.slice(-10), null, 2) });
          return;
        }

        writeResp(id, { ok: true, output: 'No metering data yet.' });
        break;
      }

      // ── v1.5.0: Room Members ─────────────────────────────────────────
      case 'room_members': {
        const roomId = args.room_id;
        if (!roomId) { writeResp(id, { ok: false, output: 'room_id is required' }); return; }
        const room = rooms.get(roomId);
        if (!room) { writeResp(id, { ok: false, output: 'Not in room: ' + roomId }); return; }

        const members = room.members || [];
        if (members.length === 0) {
          writeResp(id, { ok: true, output: 'No members tracked in room: ' + roomId });
          return;
        }
        const lines = ['Members in ' + roomId + ' (' + members.length + '):'];
        for (const m of members) {
          const name = m.agentName || 'unknown';
          const type = m.agentType || '?';
          const caps = m.capabilities?.join(', ') || 'none';
          const status = m.status || 'active';
          lines.push('  ' + m.peerId.slice(0, 16) + '... ' + name + ' (' + type + ') [' + status + '] caps: ' + caps);
        }
        writeResp(id, { ok: true, output: lines.join('\n') });
        break;
      }

      // ── COHERE distributed inference toggle ─────────────────────────
      case 'cohere_enable': {
        cohereActive = true;
        dlog('COHERE query handler enabled — listening on nexus.cohere.query');
        // WO-1.5: Publish capacity announcement on enable
        if (typeof _publishCapacityAnnouncement === 'function') {
          try { _publishCapacityAnnouncement(); } catch {}
        }
        writeResp(id, { ok: true, output: 'COHERE inference handler enabled' });
        break;
      }
      case 'cohere_disable': {
        cohereActive = false;
        dlog('COHERE query handler disabled');
        writeResp(id, { ok: true, output: 'COHERE inference handler disabled' });
        break;
      }

      // ── COHERE stats ───────────────────────────────────────────────
      case 'cohere_stats': {
        var _csUptime = Math.round((Date.now() - _cohereStats.startedAt) / 1000);
        var _csAvgLatency = _cohereStats.queriesAnswered > 0 ? Math.round(_cohereStats.totalLatencyMs / _cohereStats.queriesAnswered) : 0;
        var _csModels = Object.entries(_cohereStats.modelsUsed).sort(function(a, b) { return b[1] - a[1]; });
        var _csPeers = Object.entries(_cohereStats.peersServed).sort(function(a, b) { return b[1] - a[1]; });
        var _csLines = [
          '═══ COHERE Network Stats ═══',
          '',
          'Status: ' + (cohereActive ? 'ACTIVE' : 'INACTIVE'),
          'Daemon PID: ' + process.pid,
          'Uptime: ' + Math.floor(_csUptime / 3600) + 'h ' + Math.floor((_csUptime % 3600) / 60) + 'm ' + (_csUptime % 60) + 's',
          'Last query: ' + (_cohereStats.lastQueryAt ? new Date(_cohereStats.lastQueryAt).toISOString() : 'never'),
          '',
          '── Queries ──',
          '  Received:  ' + _cohereStats.queriesReceived,
          '  Answered:  ' + _cohereStats.queriesAnswered,
          '  Errors:    ' + _cohereStats.queriesErrors,
          '  Sent out:  ' + _cohereStats.queriesSent,
          '  Avg latency: ' + _csAvgLatency + 'ms',
          '',
          '── Data ──',
          '  Bytes in:  ' + (_cohereStats.bytesIn / 1024).toFixed(1) + ' KB',
          '  Bytes out: ' + (_cohereStats.bytesOut / 1024).toFixed(1) + ' KB',
          '',
          '── Models Used ──',
        ];
        if (_csModels.length === 0) _csLines.push('  (none yet)');
        for (var _cmi = 0; _cmi < _csModels.length; _cmi++) {
          _csLines.push('  ' + _csModels[_cmi][0] + ': ' + _csModels[_cmi][1] + ' queries');
        }
        _csLines.push('');
        _csLines.push('── Peers Served ──');
        if (_csPeers.length === 0) _csLines.push('  (none yet)');
        for (var _cpi = 0; _cpi < _csPeers.length; _cpi++) {
          _csLines.push('  ' + _csPeers[_cpi][0].slice(0, 20) + '...: ' + _csPeers[_cpi][1] + ' queries');
        }
        _csLines.push('');
        _csLines.push('── Model Allowlist ──');
        if (!_cohereAllowedModels) {
          _csLines.push('  All downloaded models exposed (no filter)');
        } else {
          _csLines.push('  ' + [..._cohereAllowedModels].join(', '));
        }
        writeResp(id, { ok: true, output: _csLines.join('\n') });
        break;
      }

      // ── COHERE model exposure control ──────────────────────────────
      case 'cohere_allow_model': {
        var _camModel = (args.model || '').trim();
        if (!_camModel) { writeResp(id, { ok: false, output: 'model name required' }); break; }
        if (!_cohereAllowedModels) _cohereAllowedModels = new Set();
        _cohereAllowedModels.add(_camModel);
        _saveModelAllowlist();
        writeResp(id, { ok: true, output: 'Model allowed for COHERE: ' + _camModel + ' (total: ' + _cohereAllowedModels.size + ')' });
        break;
      }
      case 'cohere_deny_model': {
        var _cdmModel = (args.model || '').trim();
        if (!_cdmModel) { writeResp(id, { ok: false, output: 'model name required' }); break; }
        if (_cohereAllowedModels) {
          _cohereAllowedModels.delete(_cdmModel);
          if (_cohereAllowedModels.size === 0) _cohereAllowedModels = null; // empty = all allowed
          _saveModelAllowlist();
        }
        writeResp(id, { ok: true, output: 'Model denied for COHERE: ' + _cdmModel + (_cohereAllowedModels ? ' (remaining: ' + _cohereAllowedModels.size + ')' : ' (all models now exposed)') });
        break;
      }
      case 'cohere_list_models': {
        var _clmOllamaUrl = process.env.OLLAMA_HOST || 'http://localhost:11434';
        var _clmModels = [];
        try {
          var _clmResp = await fetch(_clmOllamaUrl + '/api/tags');
          var _clmData = await _clmResp.json();
          _clmModels = (_clmData.models || []).map(function(m) { return { name: m.name, size: m.size || 0, family: m.details?.family || '' }; });
        } catch {}
        var _clmLines = ['── Downloaded Models ──'];
        for (var _clmi = 0; _clmi < _clmModels.length; _clmi++) {
          var _clmM = _clmModels[_clmi];
          var _clmAllowed = !_cohereAllowedModels || _cohereAllowedModels.has(_clmM.name);
          var _clmSizeGB = (_clmM.size / (1024*1024*1024)).toFixed(1);
          _clmLines.push('  ' + (_clmAllowed ? '[EXPOSED]' : '[HIDDEN] ') + ' ' + _clmM.name + ' (' + _clmSizeGB + 'GB' + (_clmM.family ? ', ' + _clmM.family : '') + ')');
        }
        if (_clmModels.length === 0) _clmLines.push('  (no models found — is Ollama running?)');
        _clmLines.push('');
        _clmLines.push(_cohereAllowedModels ? 'Allowlist: ' + [..._cohereAllowedModels].join(', ') : 'Allowlist: ALL (no filter active)');
        writeResp(id, { ok: true, output: _clmLines.join('\n') });
        break;
      }

      // ── IPFS publish ───────────────────────────────────────────────
      case 'ipfs_add': {
        var _iaContent = args.content || '';
        if (!_iaContent) { writeResp(id, { ok: false, output: 'content required' }); break; }
        var _iaReady = await _ensureHelia();
        if (!_iaReady || !_heliaFs) {
          // Fallback: SHA-256 hash as pseudo-CID
          var _iaCrypto = await import('node:crypto');
          var _iaHash = _iaCrypto.createHash('sha256').update(_iaContent).digest('hex');
          var _iaFallbackDir = join(nexusDir, 'ipfs', 'local');
          mkdirSync(_iaFallbackDir, { recursive: true });
          writeFileSync(join(_iaFallbackDir, _iaHash + '.json'), _iaContent);
          writeResp(id, { ok: true, output: JSON.stringify({ cid: 'sha256:' + _iaHash, pinned: true, backend: 'local-fs', path: join(_iaFallbackDir, _iaHash + '.json') }) });
          break;
        }
        try {
          var _iaEncoder = new TextEncoder();
          var _iaCid = await _heliaFs.addBytes(_iaEncoder.encode(_iaContent));
          // Pin locally (Helia pins by default on add, but explicit for clarity)
          for await (var _p of _heliaNode.pins.add(_iaCid)) { /* pinning */ }
          writeResp(id, { ok: true, output: JSON.stringify({ cid: _iaCid.toString(), pinned: true, backend: 'helia-ipfs' }) });
          dlog('IPFS: added ' + _iaCid.toString() + ' (' + _iaContent.length + ' bytes)');
        } catch (_iaErr) {
          writeResp(id, { ok: false, output: 'IPFS add error: ' + (_iaErr.message || _iaErr) });
        }
        break;
      }

      // ── IPFS pin external CID ──────────────────────────────────────
      case 'ipfs_pin': {
        var _ipCid = args.cid || '';
        if (!_ipCid) { writeResp(id, { ok: false, output: 'cid required' }); break; }
        // Validate CID format
        if (!_ipCid.startsWith('bafy') && !_ipCid.startsWith('bafk') && !_ipCid.startsWith('Qm')) {
          writeResp(id, { ok: false, output: 'Invalid CID format. Must start with bafy, bafk, or Qm' });
          break;
        }
        var _ipReady = await _ensureHelia();
        if (!_ipReady || !_heliaNode) {
          writeResp(id, { ok: false, output: 'Helia not available — cannot pin external CIDs' });
          break;
        }
        try {
          var { CID: _ipCIDClass } = await import('multiformats/cid');
          var _ipParsed = _ipCIDClass.parse(_ipCid);
          for await (var _ipPin of _heliaNode.pins.add(_ipParsed)) { /* pinning */ }
          // Register in CID registry
          var _ipRegDir = join(nexusDir, 'ipfs', 'cid-registry');
          mkdirSync(_ipRegDir, { recursive: true });
          var _ipRegFile = join(_ipRegDir, 'learning-cids.json');
          var _ipReg = {};
          try { if (existsSync(_ipRegFile)) _ipReg = JSON.parse(readFileSync(_ipRegFile, 'utf8')); } catch {}
          _ipReg['pin-' + Date.now().toString(36)] = {
            cid: _ipCid,
            source: args.source || 'manual-pin',
            pinned: true,
            timestamp: Date.now(),
          };
          writeFileSync(_ipRegFile, JSON.stringify(_ipReg, null, 2));
          dlog('IPFS: pinned external CID ' + _ipCid.slice(0, 20) + '...');
          writeResp(id, { ok: true, output: JSON.stringify({ cid: _ipCid, pinned: true }) });
        } catch (_ipErr) {
          writeResp(id, { ok: false, output: 'Pin error: ' + (_ipErr.message || _ipErr) });
        }
        break;
      }

      // ── IPFS list pinned CIDs ──────────────────────────────────────
      case 'ipfs_ls': {
        var _ilPins = [];
        // Merge from Helia pins (if available) + CID registry
        try {
          if (_heliaReady && _heliaNode) {
            for await (var _ilCid of _heliaNode.pins.ls()) {
              _ilPins.push({ cid: _ilCid.cid.toString(), source: 'helia', pinned: true });
            }
          }
        } catch {}
        // Merge CID registry metadata
        try {
          var _ilRegFile = join(nexusDir, 'ipfs', 'cid-registry', 'learning-cids.json');
          if (existsSync(_ilRegFile)) {
            var _ilReg = JSON.parse(readFileSync(_ilRegFile, 'utf8'));
            for (var _ilKey of Object.keys(_ilReg)) {
              var _ilEntry = _ilReg[_ilKey];
              if (!_ilPins.some(function(p) { return p.cid === _ilEntry.cid; })) {
                _ilPins.push({ cid: _ilEntry.cid, source: _ilEntry.source, pinned: _ilEntry.pinned, timestamp: _ilEntry.timestamp });
              }
            }
          }
        } catch {}
        // Also include identity CIDs
        try {
          var _ilIdFile = join(nexusDir, '..', 'identity', 'cids.json');
          if (existsSync(_ilIdFile)) {
            var _ilIdCids = JSON.parse(readFileSync(_ilIdFile, 'utf8'));
            for (var _ilIdKey of Object.keys(_ilIdCids)) {
              var _ilIdCid = _ilIdCids[_ilIdKey];
              if (_ilIdCid && !_ilPins.some(function(p) { return p.cid === _ilIdCid; })) {
                _ilPins.push({ cid: _ilIdCid, source: 'identity-' + _ilIdKey, pinned: true });
              }
            }
          }
        } catch {}
        writeResp(id, { ok: true, output: JSON.stringify({ pins: _ilPins, count: _ilPins.length }) });
        break;
      }

      // ── COHERE distributed learning — publish insight to mesh ──────
      case 'cohere_publish_insight': {
        if (!cohereActive) { writeResp(id, { ok: false, output: 'COHERE not active' }); break; }
        if (!_natsConn || !_natsCodec) { writeResp(id, { ok: false, output: 'NATS not connected' }); break; }
        var _cpiInsight = args.insight || '';
        var _cpiCategory = args.category || 'strategy';
        var _cpiConfidence = parseFloat(args.confidence || '0.7');
        if (!_cpiInsight) { writeResp(id, { ok: false, output: 'insight text required' }); break; }
        try {
          // Detect local model tier from warm model
          var _cpiTier = 'unknown';
          try {
            var _cpiOllamaUrl = process.env.OLLAMA_HOST || 'http://localhost:11434';
            if (_cLastModel) {
              var _cpiTagsResp = await fetch(_cpiOllamaUrl + '/api/tags');
              var _cpiTags = await _cpiTagsResp.json();
              var _cpiModel = (_cpiTags.models || []).find(function(m) { return m.name === _cLastModel; });
              if (_cpiModel) {
                var _cpiSizeGB = (_cpiModel.size || 0) / (1024*1024*1024);
                _cpiTier = _cpiSizeGB >= 20 ? 'large' : _cpiSizeGB >= 5 ? 'medium' : 'small';
              }
            }
          } catch {}
          // WO-DL4: IPFS pin the insight content for persistence + cross-node retrieval
          var _cpiCid = null;
          try {
            var _cpiHeliaReady = await _ensureHelia();
            if (_cpiHeliaReady && _heliaFs) {
              var _cpiBytes = new TextEncoder().encode(JSON.stringify({ insight: _cpiInsight, category: _cpiCategory, confidence: _cpiConfidence, tier: _cpiTier, agent: agentName, ts: Date.now() }));
              var _cpiCidObj = await _heliaFs.addBytes(_cpiBytes);
              for await (var _cpiPin of _heliaNode.pins.add(_cpiCidObj)) {}
              _cpiCid = _cpiCidObj.toString();
              dlog('IPFS pinned insight: ' + _cpiCid);
            }
          } catch (_cpiIpfsErr) {
            dlog('IPFS pin failed (graceful): ' + (_cpiIpfsErr.message || _cpiIpfsErr));
          }

          var _cpiDelta = {
            type: 'cohere.learning',
            delta_id: 'insight-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 6),
            source_peer: nexus.peerId,
            source_agent: agentName,
            insight: String(_cpiInsight).slice(0, 500),
            category: _cpiCategory,
            confidence: _cpiConfidence,
            model_tier: _cpiTier,
            cid: _cpiCid,  // null if IPFS unavailable, CID string if pinned
            timestamp: Date.now(),
          };
          _natsConn.publish('nexus.cohere.learning', _natsCodec.encode(JSON.stringify(_cpiDelta)));
          _cohereStats.queriesSent = (_cohereStats.queriesSent || 0) + 1;
          _saveStats();
          dlog('COHERE learning published: ' + _cpiCategory + ' — ' + String(_cpiInsight).slice(0, 80));
          writeResp(id, { ok: true, output: 'Insight published to COHERE mesh: ' + _cpiDelta.delta_id });
        } catch (_cpiErr) {
          writeResp(id, { ok: false, output: 'Publish error: ' + (_cpiErr.message || _cpiErr) });
        }
        break;
      }

      // ── Metered Inference Exposure ────────────────────────────────────
      case 'expose': {
        // Clear old inference capabilities before registering new ones
        // (prevents stale local Ollama models when switching to passthrough, or vice versa)
        if (typeof nexus.getRegisteredCapabilities === 'function' && typeof nexus.unregisterCapability === 'function') {
          var oldCaps = nexus.getRegisteredCapabilities();
          for (var oci = 0; oci < oldCaps.length; oci++) {
            if (oldCaps[oci].startsWith('inference:') || oldCaps[oci] === 'system_metrics' || oldCaps[oci] === '__list_capabilities') {
              try { nexus.unregisterCapability(oldCaps[oci]); } catch {}
            }
          }
          dlog('expose: cleared ' + oldCaps.length + ' old capabilities');
        }

        // Auth key for gating inference access (passed from ExposeP2PGateway)
        var exposeAuthKey = args.auth_key || '';
        if (exposeAuthKey) {
          dlog('expose: auth key configured (' + exposeAuthKey.length + ' chars)');
        }

        // Passthrough mode: forward from a remote /endpoint (Chutes, Groq, etc.)
        var isPassthrough = args.passthrough === 'true';
        var endpointAuth = args.endpoint_auth || '';
        if (isPassthrough) {
          dlog('expose: PASSTHROUGH mode — forwarding from upstream endpoint');
        }

        // Query models from the backend
        var rawUrl = args.ollama_url || process.env.OLLAMA_URL || 'http://localhost:11434';
        // For passthrough: normalize URL — strip /v1/chat/completions, /v1, etc. so we can append our own paths
        const ollamaUrl = isPassthrough
          ? rawUrl.replace(/\/+$/, '').replace(/\/chat\/completions$/, '').replace(/\/completions$/, '').replace(/\/models(\/.*)?$/, '').replace(/\/v1$/, '').replace(/\/+$/, '')
          : rawUrl;
        let models = [];

        if (isPassthrough) {
          // Passthrough: query /v1/models from the upstream endpoint (OpenAI-compatible)
          try {
            var modelsHeaders = { 'Content-Type': 'application/json' };
            if (endpointAuth) modelsHeaders['Authorization'] = 'Bearer ' + endpointAuth;
            var modelsUrl = ollamaUrl.replace(/\/+$/, '') + '/v1/models';
            dlog('expose: passthrough querying models from ' + modelsUrl);
            const modelsResp = await fetch(modelsUrl, { headers: modelsHeaders });
            if (modelsResp.ok) {
              const modelsData = await modelsResp.json();
              var modelList = modelsData.data || modelsData.models || [];
              models = modelList.map(function(m) {
                return {
                  name: m.id || m.name || 'unknown',
                  size: 0,
                  parameterSize: '',
                  family: m.owned_by || '',
                  quantization: '',
                };
              });
            } else {
              var errText = '';
              try { errText = await modelsResp.text(); } catch {}
              writeResp(id, { ok: false, output: 'Upstream /v1/models returned ' + modelsResp.status + ': ' + errText.slice(0, 200) });
              return;
            }
          } catch (e) {
            writeResp(id, { ok: false, output: 'Cannot reach upstream endpoint at ' + ollamaUrl + ': ' + e.message });
            return;
          }
        } else {
          // Local Ollama: query /api/tags
          try {
            const tagsResp = await fetch(ollamaUrl + '/api/tags');
            if (tagsResp.ok) {
              const tagsData = await tagsResp.json();
              models = (tagsData.models || []).map(function(m) {
                return {
                  name: m.name,
                  size: m.size || 0,
                  parameterSize: m.details ? m.details.parameter_size || '' : '',
                  family: m.details ? m.details.family || '' : '',
                  quantization: m.details ? m.details.quantization_level || '' : '',
                };
              });
            }
          } catch (e) {
            writeResp(id, { ok: false, output: 'Cannot reach Ollama at ' + ollamaUrl + ': ' + e.message });
            return;
          }
        }

        if (models.length === 0) {
          writeResp(id, { ok: false, output: isPassthrough ? 'No models found on upstream endpoint.' : 'No models found on Ollama. Pull a model first.' });
          return;
        }
        dlog('expose: found ' + models.length + ' models' + (isPassthrough ? ' (passthrough)' : ''));

        // Fetch market rates from OpenRouter (free, no auth)
        let marketRates = {};
        try {
          const orResp = await fetch('https://openrouter.ai/api/v1/models');
          if (orResp.ok) {
            const orData = await orResp.json();
            for (const m of (orData.data || [])) {
              if (m.pricing) {
                marketRates[m.id] = {
                  input: parseFloat(m.pricing.prompt || '0') * 1_000_000,
                  output: parseFloat(m.pricing.completion || '0') * 1_000_000,
                };
              }
            }
          }
        } catch { /* offline — use zero rates */ }

        // Build pricing menu — match local models to market rates
        const margin = parseFloat(args.margin || '0'); // default free — set margin > 0 to enable x402 pricing
        const pricingMenu = [];

        for (const model of models) {
          // Try to match to OpenRouter model ID
          const baseName = model.name.replace(/:latest$/, '').toLowerCase();
          let matched = null;
          for (const [orId, rates] of Object.entries(marketRates)) {
            const orBase = orId.toLowerCase();
            if (orBase.includes(baseName) || baseName.includes(orBase.split('/').pop())) {
              matched = { orId, ...rates };
              break;
            }
          }

          const entry = {
            model: model.name,
            parameterSize: model.parameterSize,
            family: model.family,
            quantization: model.quantization,
            pricing: {
              input_per_1m_tokens: matched ? +(matched.input * margin).toFixed(4) : 0,
              output_per_1m_tokens: matched ? +(matched.output * margin).toFixed(4) : 0,
              currency: 'USD',
              source: matched ? 'openrouter:' + matched.orId : 'self-hosted:free',
              margin: margin,
            },
          };
          pricingMenu.push(entry);

          // Register as nexus capability
          const capName = 'inference:' + model.name.replace(/[^a-zA-Z0-9._-]/g, '_');
          if (typeof nexus.registerCapability === 'function') {
            var capOpts = {};
            if (margin > 0 && entry.pricing.input_per_1m_tokens > 0) {
              var tokensPerReq = 1000;
              var amountPerReq = Math.round(entry.pricing.input_per_1m_tokens * tokensPerReq);
              if (amountPerReq > 0) {
                capOpts.pricing = {
                  amount: String(amountPerReq),
                  currency: 'USDC',
                  description: 'Inference: ' + model.name,
                };
              }
            }
            nexus.registerCapability(capName, async (request, stream) => {
              const logEntry = {
                ts: Date.now(),
                from: request.from || 'unknown',
                capability: capName,
                model: model.name,
                requestId: request.requestId,
              };
              const logFile = join(invocationsDir, Date.now() + '-' + capName + '.json');
              try { writeFileSync(logFile, JSON.stringify(logEntry, null, 2)); } catch {}

              // Safe stream write — consumer may have timed out and closed the stream
              var streamClosed = false;
              async function swrite(msg) {
                if (streamClosed) return;
                try { await stream.write(msg); } catch (swErr) {
                  dlog('stream.write failed (consumer likely timed out): ' + (swErr.message || swErr));
                  streamClosed = true;
                }
              }

              // Collect input via stream data events
              // NOTE: auth_key arrives in invoke.chunk data, NOT in invoke.open (request).
              // We must accept + read data FIRST, then validate auth.
              let prompt = '';
              var dataChunks = [];
              var inputDone = false;
              stream.onData((msg) => {
                if (msg.type === 'invoke.chunk') {
                  var chunk = typeof msg.data === 'string' ? msg.data : JSON.stringify(msg.data);
                  dataChunks.push(chunk);
                }
                if (msg.type === 'invoke.done' || msg.type === 'invoke.end' || msg.type === 'invoke.close') {
                  inputDone = true;
                }
              });

              // Accept the invocation so the consumer sends data
              await swrite({
                type: 'invoke.accept', version: 1,
                requestId: request.requestId, accepted: true,
              });

              // Wait for input data to arrive (invoke protocol sends data after accept)
              var waitMs = 0;
              while (!inputDone && dataChunks.length === 0 && waitMs < 5000) {
                await new Promise(function(r) { setTimeout(r, 10); });
                waitMs += 10;
              }
              prompt = dataChunks.join('');
              dlog('expose: received ' + dataChunks.length + ' chunks, prompt_len=' + prompt.length + ' inputDone=' + inputDone);

              // Auth key validation — check AFTER data is received.
              // Auth key is in the invoke.chunk data payload, not in invoke.open.
              if (exposeAuthKey) {
                var reqAuthKey = '';
                // Try extracting auth_key from parsed data payload
                try {
                  var authCheckData = JSON.parse(prompt);
                  if (authCheckData && typeof authCheckData === 'object' && authCheckData.auth_key) {
                    reqAuthKey = authCheckData.auth_key;
                  }
                } catch (authParseErr) { /* not JSON or no auth_key field */ }
                // Fallback: check invoke.open metadata (future-proofing)
                if (!reqAuthKey && request.metadata && request.metadata.auth_key) {
                  reqAuthKey = request.metadata.auth_key;
                }
                if (reqAuthKey !== exposeAuthKey) {
                  dlog('expose: auth REJECTED for ' + capName + ' from ' + (request.from || 'unknown'));
                  await swrite({ type: 'invoke.event', version: 1, requestId: request.requestId, seq: 0, event: 'error', data: 'Unauthorized — invalid or missing auth key' });
                  await swrite({ type: 'invoke.done', version: 1, requestId: request.requestId, usage: { inputBytes: 0, outputBytes: 0 } });
                  try { stream.close(); } catch {}
                  return;
                }
                dlog('expose: auth OK for ' + capName);
              }

              // Forward to Ollama — supports both flat prompt and structured messages
              try {
                var parsedReq = null;
                try { parsedReq = JSON.parse(prompt); } catch (pe) { dlog('expose: JSON parse error: ' + (pe.message || pe)); }

                var genResp, genData, output, inputTokens, outputTokens, responsePayload;

                // Detect if requester wants streaming (outputMode from invoke.open)
                var wantsStream = request.outputMode === 'stream';

                if (parsedReq && parsedReq.messages && Array.isArray(parsedReq.messages)) {
                  // Structured request — use /v1/chat/completions (supports tools + multi-turn)
                  var chatBody = {
                    model: model.name,
                    messages: parsedReq.messages,
                    stream: wantsStream,
                  };
                  if (parsedReq.tools && parsedReq.tools.length > 0) {
                    chatBody.tools = parsedReq.tools;
                  }
                  if (parsedReq.temperature !== undefined) chatBody.temperature = parsedReq.temperature;
                  if (parsedReq.max_tokens) chatBody.max_tokens = parsedReq.max_tokens;
                  if (parsedReq.think !== undefined) chatBody.think = parsedReq.think;

                  var chatHeaders = { 'Content-Type': 'application/json' };
                  if (isPassthrough && endpointAuth) chatHeaders['Authorization'] = 'Bearer ' + endpointAuth;
                  dlog('expose: calling /v1/chat/completions with ' + chatBody.messages.length + ' messages, tools=' + (chatBody.tools ? chatBody.tools.length : 0) + ' stream=' + wantsStream + (isPassthrough ? ' (passthrough)' : ''));
                  genResp = await fetch(ollamaUrl + '/v1/chat/completions', {
                    method: 'POST',
                    headers: chatHeaders,
                    body: JSON.stringify(chatBody),
                    signal: AbortSignal.timeout(210000),
                  });

                  if (wantsStream && genResp.ok && genResp.body) {
                    // ── SSE streaming path — read tokens and send as invoke.event ──
                    var sseReader = genResp.body.getReader();
                    var sseDecoder = new TextDecoder();
                    var sseBuf = '';
                    var sseSeq = 0;
                    var sseContent = '';
                    var sseToolCalls = [];
                    inputTokens = 0;
                    outputTokens = 0;

                    while (true) {
                      var sseChunk = await sseReader.read();
                      if (sseChunk.done) break;
                      sseBuf += sseDecoder.decode(sseChunk.value, { stream: true });
                      var sseLines = sseBuf.split('\n');
                      sseBuf = sseLines.pop();

                      for (var sli = 0; sli < sseLines.length; sli++) {
                        var sseLine = sseLines[sli];
                        if (!sseLine.startsWith('data: ')) continue;
                        var sseData = sseLine.slice(6).trim();
                        if (sseData === '[DONE]') continue;

                        try {
                          var sseObj = JSON.parse(sseData);
                          var sseDelta = (sseObj.choices && sseObj.choices[0] && sseObj.choices[0].delta) || {};
                          var sseToken = sseDelta.content || '';
                          if (sseToken) {
                            sseContent += sseToken;
                            await swrite({
                              type: 'invoke.event', version: 1,
                              requestId: request.requestId, seq: sseSeq++,
                              event: 'token', data: sseToken,
                            });
                          }
                          // Collect tool calls from streaming deltas
                          if (sseDelta.tool_calls) {
                            for (var stci = 0; stci < sseDelta.tool_calls.length; stci++) {
                              var stc = sseDelta.tool_calls[stci];
                              if (stc.index !== undefined) {
                                while (sseToolCalls.length <= stc.index) sseToolCalls.push({ id: '', function: { name: '', arguments: '' } });
                                if (stc.id) sseToolCalls[stc.index].id = stc.id;
                                if (stc.function) {
                                  if (stc.function.name) sseToolCalls[stc.index].function.name = stc.function.name;
                                  if (stc.function.arguments) sseToolCalls[stc.index].function.arguments += stc.function.arguments;
                                }
                              }
                            }
                          }
                          // Capture usage from final chunk
                          if (sseObj.usage) {
                            inputTokens = sseObj.usage.prompt_tokens || 0;
                            outputTokens = sseObj.usage.completion_tokens || 0;
                          }
                        } catch (sseParseErr) { /* skip bad SSE lines */ }
                      }
                    }

                    // Strip thinking tags from accumulated content
                    output = sseContent.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
                    // Build final response payload for the result event
                    var sseChoices = [{
                      message: {
                        content: output,
                        tool_calls: sseToolCalls.length > 0 ? sseToolCalls : undefined,
                      },
                    }];
                    responsePayload = {
                      model: model.name,
                      response: output,
                      choices: sseChoices,
                      usage: { input_tokens: inputTokens, output_tokens: outputTokens },
                      pricing: entry.pricing,
                    };
                  } else {
                    // ── Unary path (stream: false or streaming not available) ──
                    genData = await genResp.json();
                    dlog('expose: ollama response keys=' + Object.keys(genData).join(',') + ' choices=' + (genData.choices ? genData.choices.length : 0));

                    var chatChoices = genData.choices || [];
                    var firstMsg = (chatChoices[0] || {}).message || {};
                    var rawContent = firstMsg.content || '';
                    output = rawContent.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
                    if (!output && firstMsg.reasoning) {
                      output = firstMsg.reasoning;
                    }
                    if (chatChoices[0] && chatChoices[0].message) {
                      chatChoices[0].message.content = output;
                    }
                    dlog('expose: rawContent_len=' + rawContent.length + ' output_len=' + output.length);
                    inputTokens = (genData.usage || {}).prompt_tokens || 0;
                    outputTokens = (genData.usage || {}).completion_tokens || 0;

                    responsePayload = {
                      model: model.name,
                      response: output,
                      choices: chatChoices,
                      usage: { input_tokens: inputTokens, output_tokens: outputTokens },
                      pricing: entry.pricing,
                    };
                  }
                } else if (isPassthrough) {
                  // Passthrough flat prompt — convert to /v1/chat/completions (upstream has no /api/generate)
                  var flatPrompt = parsedReq && parsedReq.prompt ? parsedReq.prompt : (prompt || 'Hello');
                  var ptHeaders = { 'Content-Type': 'application/json' };
                  if (endpointAuth) ptHeaders['Authorization'] = 'Bearer ' + endpointAuth;
                  genResp = await fetch(ollamaUrl + '/v1/chat/completions', {
                    method: 'POST',
                    headers: ptHeaders,
                    body: JSON.stringify({
                      model: model.name,
                      messages: [{ role: 'user', content: flatPrompt }],
                      stream: false,
                    }),
                    signal: AbortSignal.timeout(210000), // 3.5 min
                  });
                  genData = await genResp.json();
                  var ptChoices = genData.choices || [];
                  var ptMsg = (ptChoices[0] || {}).message || {};
                  output = (ptMsg.content || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();
                  inputTokens = (genData.usage || {}).prompt_tokens || 0;
                  outputTokens = (genData.usage || {}).completion_tokens || 0;

                  responsePayload = {
                    model: model.name,
                    response: output,
                    choices: ptChoices,
                    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
                    pricing: entry.pricing,
                  };
                } else {
                  // Flat prompt — use /api/generate (local Ollama)
                  var flatPrompt = parsedReq && parsedReq.prompt ? parsedReq.prompt : (prompt || 'Hello');
                  genResp = await fetch(ollamaUrl + '/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      model: model.name,
                      prompt: flatPrompt,
                      stream: false,
                    }),
                    signal: AbortSignal.timeout(210000), // 3.5 min
                  });
                  genData = await genResp.json();
                  output = genData.response || '';
                  inputTokens = genData.prompt_eval_count || 0;
                  outputTokens = genData.eval_count || 0;

                  responsePayload = {
                    model: model.name,
                    response: output,
                    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
                    pricing: entry.pricing,
                  };
                }

                // Attach system metrics to response — clients get CPU/GPU/RAM
                // for free without a separate invoke_capability round-trip
                try {
                  var _sm = await _collectSysMetrics();
                  if (_sm) responsePayload.system = _sm;
                } catch {}

                // Stream result back
                await swrite({
                  type: 'invoke.event', version: 1,
                  requestId: request.requestId, seq: 0,
                  event: 'result',
                  data: JSON.stringify(responsePayload),
                });

                await swrite({
                  type: 'invoke.done', version: 1,
                  requestId: request.requestId,
                  usage: {
                    inputBytes: prompt.length,
                    outputBytes: (output || '').length,
                    tokens: inputTokens + outputTokens,
                  },
                });

                // Stash token data for the metering hook (which fires after handler returns
                // and has the correct peerId from connection context)
                _tokensByRequest[request.requestId] = {
                  model: model.name,
                  inputBytes: prompt.length,
                  outputBytes: (output || '').length,
                  tokens: inputTokens + outputTokens,
                  inputTokens: inputTokens,
                  outputTokens: outputTokens,
                };
              } catch (e) {
                dlog('expose: inference error: ' + (e.message || e));
                await swrite({
                  type: 'invoke.event', version: 1,
                  requestId: request.requestId, seq: 0,
                  event: 'error', data: 'Inference failed: ' + e.message,
                });
                await swrite({
                  type: 'invoke.done', version: 1,
                  requestId: request.requestId,
                  usage: { inputBytes: 0, outputBytes: 0 },
                });
              }
              try { stream.close(); } catch {}
            }, capOpts);
          }
        }

        // Register system_metrics capability — returns CPU/GPU/memory utilization
        if (typeof nexus.registerCapability === 'function') {
          nexus.registerCapability('system_metrics', async (request, stream) => {
            // Stream safety wrapper — prevents unguarded writes after consumer disconnects
            var smStreamClosed = false;
            async function smWrite(msg) {
              if (smStreamClosed) return;
              try { await stream.write(msg); } catch { smStreamClosed = true; }
            }
            // Collect input via stream data events (auth_key arrives in invoke.chunk, NOT invoke.open)
            var smDataChunks = [];
            var smInputDone = false;
            stream.onData(function(msg) {
              if (msg.type === 'invoke.chunk') {
                smDataChunks.push(typeof msg.data === 'string' ? msg.data : JSON.stringify(msg.data));
              }
              if (msg.type === 'invoke.done' || msg.type === 'invoke.end') {
                smInputDone = true;
              }
            });

            // Accept invocation so consumer sends data
            await smWrite({ type: 'invoke.accept', version: 1, requestId: request.requestId, accepted: true });

            // Wait briefly for data (auth key arrives in chunk)
            var smWait = 0;
            while (!smInputDone && smDataChunks.length === 0 && smWait < 2000) {
              await new Promise(function(r) { setTimeout(r, 10); });
              smWait += 10;
            }

            // Auth check — extract auth_key from chunk data
            if (exposeAuthKey) {
              var smAuthKey = '';
              var smPayload = smDataChunks.join('');
              try {
                var smParsed = JSON.parse(smPayload);
                if (smParsed && typeof smParsed === 'object' && smParsed.auth_key) {
                  smAuthKey = smParsed.auth_key;
                }
              } catch {}
              // Fallback: check invoke.open metadata (future-proofing)
              if (!smAuthKey && request.metadata && request.metadata.auth_key) {
                smAuthKey = request.metadata.auth_key;
              }
              if (smAuthKey !== exposeAuthKey) {
                dlog('system_metrics: auth REJECTED from ' + (request.from || 'unknown'));
                await smWrite({ type: 'invoke.event', version: 1, requestId: request.requestId, seq: 0, event: 'error', data: 'Unauthorized' });
                await smWrite({ type: 'invoke.done', version: 1, requestId: request.requestId, usage: { inputBytes: 0, outputBytes: 0 } });
                try { stream.close(); } catch {}
                return;
              }
              dlog('system_metrics: auth OK');
            }
            try {
              var os = await import('node:os');
              var loads = os.loadavg();
              var cores = os.cpus().length;
              var totalMem = os.totalmem();
              var freeMem = os.freemem();
              var usedMem = totalMem - freeMem;
              var gpuInfo = { available: false, name: '', utilization: 0, vramUsedMB: 0, vramTotalMB: 0, vramUtilization: 0 };
              try {
                var cp = await import('node:child_process');
                var smiOut = cp.execSync('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,name --format=csv,noheader,nounits 2>/dev/null', { encoding: 'utf8', timeout: 3000 });
                var smiLine = smiOut.trim().split('\n')[0];
                if (smiLine) {
                  var sp = smiLine.split(',').map(function(s) { return s.trim(); });
                  gpuInfo.available = true;
                  gpuInfo.utilization = parseInt(sp[0] || '0', 10) || 0;
                  gpuInfo.vramUsedMB = parseInt(sp[1] || '0', 10) || 0;
                  gpuInfo.vramTotalMB = parseInt(sp[2] || '0', 10) || 0;
                  gpuInfo.name = sp[3] || '';
                  gpuInfo.vramUtilization = gpuInfo.vramTotalMB > 0 ? Math.round((gpuInfo.vramUsedMB / gpuInfo.vramTotalMB) * 100) : 0;
                }
              } catch (ge) { /* no GPU */ }
              var cpuModel = '';
              try { var cpuInfoArr = os.cpus(); if (cpuInfoArr.length > 0) cpuModel = cpuInfoArr[0].model || ''; } catch {}
              var metricsPayload = {
                cpu: { utilization: Math.min(100, Math.round((loads[0] / cores) * 100)), cores: cores, model: cpuModel },
                memory: { utilization: Math.round((usedMem / totalMem) * 100), totalGB: Math.round((totalMem / (1024*1024*1024)) * 10) / 10, usedGB: Math.round((usedMem / (1024*1024*1024)) * 10) / 10 },
                gpu: gpuInfo,
                timestamp: new Date().toISOString(),
              };
              await smWrite({ type: 'invoke.event', version: 1, requestId: request.requestId, seq: 0, event: 'result', data: JSON.stringify(metricsPayload) });
              await smWrite({ type: 'invoke.done', version: 1, requestId: request.requestId, usage: { inputBytes: 0, outputBytes: JSON.stringify(metricsPayload).length } });
            } catch (me) {
              await smWrite({ type: 'invoke.event', version: 1, requestId: request.requestId, seq: 0, event: 'error', data: 'metrics error: ' + me.message });
              await smWrite({ type: 'invoke.done', version: 1, requestId: request.requestId, usage: { inputBytes: 0, outputBytes: 0 } });
            }
            try { stream.close(); } catch {}
          });
          dlog('system_metrics capability registered');
        }

        // Register __list_capabilities — allows remote peers to discover available models
        if (typeof nexus.registerCapability === 'function') {
          nexus.registerCapability('__list_capabilities', async (request, stream) => {
            dlog('__list_capabilities invoked from ' + (request.from || 'unknown'));
            await stream.write({ type: 'invoke.accept', version: 1, requestId: request.requestId, accepted: true });
            var allCaps = typeof nexus.getRegisteredCapabilities === 'function' ? nexus.getRegisteredCapabilities() : [];
            var modelsInfo = [];
            for (var ci = 0; ci < pricingMenu.length; ci++) {
              var pm = pricingMenu[ci];
              modelsInfo.push({
                name: pm.model,
                parameterSize: pm.parameterSize || '',
                family: pm.family || '',
                quantization: pm.quantization || '',
              });
            }
            var capsPayload = JSON.stringify({ capabilities: allCaps, models: modelsInfo, agentName: agentName, peerId: nexus.peerId });
            await stream.write({ type: 'invoke.event', version: 1, requestId: request.requestId, seq: 0, event: 'result', data: capsPayload });
            await stream.write({ type: 'invoke.done', version: 1, requestId: request.requestId, usage: { inputBytes: 0, outputBytes: capsPayload.length } });
            stream.close();
          });
          dlog('__list_capabilities capability registered');
        }

        // Write pricing menu to file
        const pricingFile = join(nexusDir, 'pricing.json');
        writeFileSync(pricingFile, JSON.stringify({ updated: new Date().toISOString(), models: pricingMenu }, null, 2));
        writeStatus({ exposedModels: pricingMenu.length });

        const lines = ['Exposed ' + pricingMenu.length + ' model(s) as nexus capabilities:'];
        for (const p of pricingMenu) {
          const cost = p.pricing.input_per_1m_tokens === 0
            ? 'FREE (self-hosted)'
            : '$' + p.pricing.input_per_1m_tokens + '/$' + p.pricing.output_per_1m_tokens + ' per 1M tokens';
          lines.push('  inference:' + p.model + ' — ' + cost);
        }
        lines.push('');
        lines.push('Pricing menu saved to ' + pricingFile);
        lines.push('Market rates: ' + Object.keys(marketRates).length + ' models from OpenRouter');
        writeResp(id, { ok: true, output: lines.join('\n') });
        break;
      }

      case 'pricing_menu': {
        const pricingFile = join(nexusDir, 'pricing.json');
        if (!existsSync(pricingFile)) {
          writeResp(id, { ok: false, output: 'No pricing menu. Run expose first.' });
          return;
        }
        try {
          const menu = JSON.parse(readFileSync(pricingFile, 'utf8'));
          const lines = ['Inference Pricing Menu (updated: ' + menu.updated + ')'];
          lines.push('');
          for (const m of (menu.models || [])) {
            const cost = m.pricing.input_per_1m_tokens === 0
              ? 'FREE'
              : '$' + m.pricing.input_per_1m_tokens + ' in / $' + m.pricing.output_per_1m_tokens + ' out per 1M tokens';
            lines.push('  ' + m.model + ' (' + m.parameterSize + ', ' + m.quantization + ')');
            lines.push('    ' + cost + ' [' + m.pricing.source + ']');
          }
          writeResp(id, { ok: true, output: lines.join('\n') });
        } catch (e) {
          writeResp(id, { ok: false, output: 'Failed to read pricing menu: ' + e.message });
        }
        break;
      }

      case 'ping': {
        writeResp(id, { ok: true, output: 'pong' });
        break;
      }
      default:
        writeResp(id, { ok: false, output: 'Unknown daemon command: ' + action });
    }
  } catch (err) {
    writeResp(id, { ok: false, output: 'Error: ' + (err.message || String(err)) });
  }
}

// Command polling loop — check for cmd.json every 50ms (fast IPC)
// fs.existsSync + readFileSync costs ~0.01ms per call, negligible CPU at 50ms interval.
let lastCmdId = '';
function checkCmd() {
  try {
    if (!existsSync(cmdFile)) return;
    const raw = readFileSync(cmdFile, 'utf8');
    const cmd = JSON.parse(raw);
    if (cmd.id === lastCmdId) return; // already processed
    lastCmdId = cmd.id;
    handleCmd(cmd).catch((err) => {
      writeResp(cmd.id, { ok: false, output: 'Error: ' + (err.message || String(err)) });
    });
  } catch {}
}
setInterval(checkCmd, 50);
// Also watch for cmd.json changes for instant notification (best-effort)
try {
  fsWatch(nexusDir, { persistent: false }, function(evType, filename) {
    if (filename === 'cmd.json') checkCmd();
  });
} catch {}

// Suppress EPIPE on stdout/stderr — parent may close pipe at any time
try { process.stdout.on('error', function(e) { if (e.code !== 'EPIPE') throw e; }); } catch {}
try { process.stderr.on('error', function(e) { if (e.code !== 'EPIPE') throw e; }); } catch {}

// Crash protection — prevent unhandled errors from killing the daemon.
// EPIPE/ECONNRESET/ETIMEDOUT are transient network errors — always swallow.
// The daemon MUST stay alive through any non-fatal error.
var TRANSIENT_CODES = ['EPIPE', 'ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'EHOSTUNREACH', 'ENETUNREACH', 'EAI_AGAIN'];
function isTransientError(err) {
  if (!err) return false;
  var code = err.code || '';
  var msg = err.message || String(err);
  return TRANSIENT_CODES.some(function(c) { return code === c || msg.includes(c); });
}
process.on('uncaughtException', (err) => {
  if (isTransientError(err)) {
    dlog('Swallowed transient uncaughtException: ' + (err.code || err.message || err));
    return; // keep daemon alive
  }
  var errMsg = err && err.message ? err.message : String(err);
  try { dlog('uncaughtException (non-transient): ' + errMsg); } catch {}
  writeStatus({ error: 'uncaughtException: ' + errMsg });
  // Do NOT exit — keep daemon alive even for non-transient errors
});
process.on('unhandledRejection', (reason) => {
  if (isTransientError(reason)) {
    dlog('Swallowed transient unhandledRejection: ' + (reason?.code || reason?.message || reason));
    return;
  }
  var msg = reason instanceof Error ? reason.message : String(reason);
  try { dlog('unhandledRejection (non-transient): ' + msg); } catch {}
  // Do NOT exit — keep daemon alive
});

// Connect
(async () => {
  try {
    await nexus.connect();
    // Ensure node is started (libp2p compat)
    const node = nexus.network?.node;
    if (node && typeof node.start === 'function' && !node.isStarted?.()) {
      await node.start();
    }
    connected = true;

    // Monkey-patch node.dialProtocol AND node.dial to convert string multiaddrs
    // to proper Multiaddr objects. libp2p v3 transports call .getComponents()
    // on multiaddr objects, which fails on plain strings. open-agents-nexus
    // resolves multiaddrs as plain strings in dialPeerProtocol() and stores
    // them that way in the peer store. nexus.node is only available AFTER connect().
    var _patchNode = nexus.network ? nexus.network.node : nexus.node;
    async function _ensureMultiaddrObj(peer) {
      if (typeof peer === 'string' && peer.startsWith('/')) {
        try {
          var { multiaddr: maFromStr } = await import('@multiformats/multiaddr');
          return maFromStr(peer);
        } catch (convErr) {
          dlog('multiaddr conversion failed: ' + (convErr.message || convErr));
        }
      }
      // Handle arrays of multiaddrs (e.g. from peer store)
      if (Array.isArray(peer)) {
        try {
          var { multiaddr: maFromStr2 } = await import('@multiformats/multiaddr');
          return peer.map(function(ma) {
            if (typeof ma === 'string' && ma.startsWith('/')) return maFromStr2(ma);
            if (ma && typeof ma === 'object' && !ma.getComponents) {
              // Object with .toString() but missing getComponents — re-parse
              try { return maFromStr2(String(ma)); } catch { return ma; }
            }
            return ma;
          });
        } catch {}
      }
      // Handle multiaddr-like objects missing getComponents (stale references)
      if (peer && typeof peer === 'object' && !Array.isArray(peer) && !peer.getComponents && peer.toString) {
        try {
          var { multiaddr: maFromStr3 } = await import('@multiformats/multiaddr');
          return maFromStr3(String(peer));
        } catch {}
      }
      return peer;
    }
    if (_patchNode && typeof _patchNode.dialProtocol === 'function') {
      var _origDialProtocol = _patchNode.dialProtocol.bind(_patchNode);
      _patchNode.dialProtocol = async function patchedDialProtocol(peer, protocols, options) {
        peer = await _ensureMultiaddrObj(peer);
        return _origDialProtocol(peer, protocols, options);
      };
      dlog('patched node.dialProtocol for multiaddr string conversion');
    } else {
      dlog('WARNING: could not patch node.dialProtocol — node=' + !!_patchNode);
    }
    if (_patchNode && typeof _patchNode.dial === 'function') {
      var _origDial = _patchNode.dial.bind(_patchNode);
      _patchNode.dial = async function patchedDial(peer, options) {
        peer = await _ensureMultiaddrObj(peer);
        return _origDial(peer, options);
      };
      dlog('patched node.dial for multiaddr string conversion');
    }

    // v1.5.0: Metering hook that enriches nexus records with token counts.
    // The nexus library fires this hook AFTER handler returns, so it has the correct
    // peerId from connection context. We stash token data per requestId so the hook
    // can merge them into one complete record written to metering.jsonl.
    try {
      if (nexus.metering && typeof nexus.metering.addHook === 'function') {
        nexus.metering.addHook(function(record) {
          // Only enrich inference capability records
          if (!record.capability || !record.capability.startsWith('inference:')) return;
          var tokenData = _tokensByRequest[record.id];
          delete _tokensByRequest[record.id]; // consume
          try {
            appendFileSync(meteringFile, JSON.stringify({
              timestamp: record.timestamp || Date.now(),
              peerId: record.peerId || 'unknown',
              service: record.service || record.capability,
              capability: record.capability,
              model: tokenData ? tokenData.model : (record.capability || '').replace('inference:', ''),
              direction: record.direction || 'inbound',
              inputBytes: tokenData ? tokenData.inputBytes : (record.inputBytes || 0),
              outputBytes: tokenData ? tokenData.outputBytes : (record.outputBytes || 0),
              tokens: tokenData ? tokenData.tokens : 0,
              inputTokens: tokenData ? tokenData.inputTokens : 0,
              outputTokens: tokenData ? tokenData.outputTokens : 0,
              durationMs: record.durationMs || 0,
            }) + '\n');
          } catch {}
        });
      }
    } catch {}

    // Write payment events to ledger.jsonl
    try {
      if (nexus.metering && typeof nexus.metering.addHook === 'function') {
        nexus.metering.addHook(function(record) {
          if (!record.payment) return;
          var ledgerFile = join(nexusDir, 'ledger.jsonl');
          var entry = {
            timestamp: new Date().toISOString(),
            type: record.direction === 'inbound' ? 'earned' : 'spent',
            amount: String(record.payment.amount || 0),
            amountUsd: (Number(record.payment.amount || 0) / 1000000).toFixed(6),
            peer: record.peerId || 'unknown',
            capability: record.service || record.capability || 'unknown',
            txHash: record.payment.txHash || '',
            note: 'auto:metering',
          };
          try { appendFileSync(ledgerFile, JSON.stringify(entry) + '\n'); } catch {}
        });
      }
    } catch {}

    // Init x402 wallet if key file exists
    if (hasX402Key && nexus.x402 && typeof nexus.x402.initWallet === 'function') {
      try {
        var addr = nexus.x402.initWallet(x402KeyPath);
        dlog('x402 wallet initialized: ' + addr);
      } catch (e) {
        dlog('x402 wallet init failed: ' + (e.message || e));
      }
    }

    // v1.5.0: Client-level events for global message/DM/invoke routing
    try {
      if (typeof nexus.on === 'function') {
        nexus.on('message', ({ roomId, message }) => {
          // Already handled by per-room listener, but log globally
          console.log('[msg] ' + roomId + ' from ' + (message?.sender || '?').slice(0, 16));
          // Relay received messages to NATS for frontend visibility
          if (_natsConn && _natsCodec && message && message.sender !== nexus.peerId) {
            try {
              var _msgContent = '';
              if (message.payload && message.payload.content) _msgContent = String(message.payload.content).slice(0, 500);
              else if (typeof message.content === 'string') _msgContent = message.content.slice(0, 500);
              if (_msgContent) {
                _natsConn.publish('nexus.rooms.chat', _natsCodec.encode(JSON.stringify({
                  type: 'nexus.room.message',
                  roomId: roomId,
                  peerId: message.sender || '',
                  agentName: message.senderName || (message.sender ? message.sender.slice(0, 12) : 'anon'),
                  content: _msgContent,
                  timestamp: message.timestamp || Date.now(),
                })));
              }
            } catch {}
          }
        });
        nexus.on('dm', ({ from, content, format, messageId }) => {
          // Log DMs to inbox/dm/
          const dmDir = join(inboxDir, '_dm');
          mkdirSync(dmDir, { recursive: true });
          const entry = {
            sender: from,
            content: content || '',
            format: format || 'text/plain',
            timestamp: Date.now(),
            id: messageId,
          };
          try { writeFileSync(join(dmDir, Date.now() + '.json'), JSON.stringify(entry, null, 2)); } catch {}
          console.log('[dm] from ' + (from || '?').slice(0, 16));
        });
        nexus.on('invoke', ({ from, capability, requestId }) => {
          console.log('[invoke] ' + capability + ' from ' + (from || '?').slice(0, 16) + ' req=' + requestId);
        });
      }
    } catch {}

    // Subscribe to NATS peer announcements — cache remote peer capabilities
    var discoveredPeers = {};
    var discoveredPeersFile = join(nexusDir, 'discovered-peers.json');
    try {
      if (existsSync(discoveredPeersFile)) {
        discoveredPeers = JSON.parse(readFileSync(discoveredPeersFile, 'utf8'));
      }
    } catch {}
    try {
      if (nexus.nats && typeof nexus.nats.subscribe === 'function') {
        nexus.nats.subscribe(function(announcement) {
          if (!announcement || !announcement.peerId) return;
          if (announcement.peerId === nexus.peerId) return; // skip self
          discoveredPeers[announcement.peerId] = {
            peerId: announcement.peerId,
            agentName: announcement.agentName || '',
            capabilities: announcement.capabilities || [],
            multiaddrs: announcement.multiaddrs || [],
            lastSeen: Date.now(),
          };
          try { writeFileSync(discoveredPeersFile, JSON.stringify(discoveredPeers, null, 2)); } catch {}
          dlog('NATS peer discovered: ' + String(announcement.peerId).slice(0, 20) + ' caps=' + (announcement.capabilities || []).length);
        });
        dlog('NATS peer discovery subscription active');
      }
    } catch (natsSubErr) {
      dlog('NATS subscribe failed: ' + (natsSubErr.message || natsSubErr));
    }

    // NATS invoke relay — subscribe to nexus.invoke.<myPeerId> for cross-NAT invoke fallback.
    // When a remote peer can't direct-dial us (all multiaddrs are private), they retry
    // via NATS request/reply. We dispatch to the registered capability handler using a
    // mock stream interface, then reply with the result.
    try {
      var _nc = nexus.nats && nexus.nats.nc ? nexus.nats.nc : null;
      if (_nc) {
        var _nws = await import('nats.ws');
        _natsConn = _nc;
        _natsCodec = _nws.StringCodec();
        var _natsInvSubject = 'nexus.invoke.' + nexus.peerId;
        var _natsInvSub = _nc.subscribe(_natsInvSubject);
        dlog('NATS invoke relay listening on ' + _natsInvSubject);

        (async function _natsInvokeLoop() {
          for await (var _nm of _natsInvSub) {
            try {
              var _nReq = JSON.parse(_natsCodec.decode(_nm.data));
              var _nCap = _nReq.capability || '';
              var _nInput = _nReq.input || {};
              var _nInputStr = typeof _nInput === 'string' ? _nInput : JSON.stringify(_nInput);
              dlog('NATS relay invoke: cap=' + _nCap + ' from=' + (_nReq.from || 'unknown').slice(0, 20));

              // Look up registered handler
              var _nHandler = null;
              if (nexus.capabilityHandlers && typeof nexus.capabilityHandlers.get === 'function') {
                _nHandler = nexus.capabilityHandlers.get(_nCap);
              }
              if (!_nHandler) {
                var _nCaps = typeof nexus.getRegisteredCapabilities === 'function'
                  ? nexus.getRegisteredCapabilities() : [];
                _nm.respond(_natsCodec.encode(JSON.stringify({
                  error: 'Capability not found: ' + _nCap + '. Available: ' + _nCaps.join(', ')
                })));
                continue;
              }

              // Create mock stream handle that captures the handler's output.
              // Uses retroactive delivery: if handler registers onData after data is
              // already queued, deliver immediately. Eliminates timing race.
              var _nResults = [];
              var _nDataCbs = [];
              var _nDataDelivered = false;
              var _nHandle = {
                write: async function(msg) {
                  if (msg && msg.type === 'invoke.event') _nResults.push(msg);
                },
                onData: function(cb) {
                  if (_nDataDelivered) {
                    // Data already flushed — deliver retroactively
                    try {
                      cb({ type: 'invoke.chunk', data: _nInputStr, seq: 0 });
                      cb({ type: 'invoke.done' });
                    } catch {}
                  } else {
                    _nDataCbs.push(cb);
                  }
                },
                close: function() {},
              };
              // Flush input data to any registered onData listeners after 50ms
              setTimeout(function() {
                _nDataDelivered = true;
                for (var _di = 0; _di < _nDataCbs.length; _di++) {
                  try {
                    _nDataCbs[_di]({ type: 'invoke.chunk', data: _nInputStr, seq: 0 });
                    _nDataCbs[_di]({ type: 'invoke.done' });
                  } catch {}
                }
              }, 50);

              var _nRequest = {
                type: 'invoke.open',
                requestId: 'nats-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8),
                capability: _nCap,
                from: _nReq.from || 'nats-peer',
              };

              await _nHandler(_nRequest, _nHandle);

              // Extract result from captured invoke.event messages
              var _nResult = null;
              if (_nResults.length === 1) {
                _nResult = _nResults[0].data;
              } else if (_nResults.length > 1) {
                _nResult = _nResults.map(function(r) { return r.data; });
              }

              _nm.respond(_natsCodec.encode(JSON.stringify(_nResult !== null && _nResult !== undefined ? _nResult : { ok: true })));
              dlog('NATS relay invoke SUCCESS: ' + _nCap);
            } catch (_nErr) {
              dlog('NATS relay invoke error: ' + (_nErr.message || _nErr));
              try {
                _nm.respond(_natsCodec.encode(JSON.stringify({ error: String(_nErr.message || _nErr) })));
              } catch {}
            }
          }
        })().catch(function(e) { dlog('NATS invoke relay loop error: ' + (e.message || e)); });
      } else {
        dlog('NATS not connected — invoke relay disabled (cross-NAT invoke will not work)');
      }
    } catch (_nSetupErr) {
      dlog('NATS invoke relay setup failed: ' + (_nSetupErr.message || _nSetupErr));
    }

    // ── COHERE distributed inference handler ─────────────────────────
    // ── COHERE distributed inference handler ─────────────────────────
    // Subscribe to nexus.cohere.query, process through local Ollama,
    // publish response to nexus.cohere.response.
    // SECURITY INVARIANTS:
    //   1. Handler constructs ISOLATED messages — no history, no system prompt
    //   2. Only /api/tags (read model list) and /api/chat (inference) are called
    //   3. NEVER calls /api/pull, /api/delete, /api/push, /api/create, /api/copy
    //   4. Model allowlist filters which models are served to remote queries
    //   5. Inbound queries scanned for leaked secrets
    if (_natsConn && _natsCodec) {
      const _cohereSub = _natsConn.subscribe('nexus.cohere.query');
      (async function() {
        for await (const _cMsg of _cohereSub) {
          if (!cohereActive) continue;
          try {
            const _cData = JSON.parse(_natsCodec.decode(_cMsg.data));
            if (!_cData.queryId || !_cData.query) continue;
            // Dedup: skip if we already processed this queryId
            if (cohereDedup.has(_cData.queryId)) continue;
            cohereDedup.set(_cData.queryId, Date.now());
            // Prune dedup map (60s TTL)
            const _cNow = Date.now();
            for (const [_cK, _cV] of cohereDedup) { if (_cNow - _cV > 60000) cohereDedup.delete(_cK); }

            // Stats: track received
            _cohereStats.queriesReceived++;
            _cohereStats.lastQueryAt = Date.now();
            _cohereStats.bytesIn += (_cData.query || '').length;
            if (_cData.source) { _cohereStats.peersServed[_cData.source] = (_cohereStats.peersServed[_cData.source] || 0) + 1; }

            // WO-1.3: Multi-node claim coordination (first-claim-wins)
            // Step 1: Random jitter 0-2s to stagger nodes
            await new Promise(function(r) { setTimeout(r, Math.random() * 2000); });
            if (!cohereActive) continue;

            // Step 2: Check if another node already claimed this query
            var _cClaim = cohereClaims.get(_cData.queryId);
            if (_cClaim && _cClaim.claimedBy !== nexus.peerId) {
              dlog('COHERE skip: query ' + _cData.queryId + ' already claimed by ' + _cClaim.claimedBy.slice(0, 16));
              continue;
            }

            // Step 3: Publish our claim BEFORE starting inference
            // Other nodes will see this and skip processing
            _natsConn.publish('nexus.cohere.claimed', _natsCodec.encode(JSON.stringify({
              queryId: _cData.queryId,
              claimedBy: nexus.peerId,
              agentName: agentName,
              timestamp: Date.now(),
            })));
            cohereClaims.set(_cData.queryId, { claimedBy: nexus.peerId, timestamp: Date.now() });
            dlog('COHERE claim: ' + _cData.queryId + ' claimed by us');

            // Step 4: Brief pause to let our claim propagate (50ms)
            await new Promise(function(r) { setTimeout(r, 50); });

            // Step 5: Re-check — if someone else claimed during our jitter, yield
            var _cReclaim = cohereClaims.get(_cData.queryId);
            if (_cReclaim && _cReclaim.claimedBy !== nexus.peerId) {
              dlog('COHERE yield: query ' + _cData.queryId + ' claimed by ' + _cReclaim.claimedBy.slice(0, 16) + ' during propagation');
              continue;
            }

            dlog('COHERE query: ' + _cData.queryId + ' — ' + (_cData.query || '').slice(0, 80));
            const _cStart = Date.now();

            // OLLAMA SAFETY: Only two endpoints are ever called:
            //   GET  /api/tags  — read available models (READ-ONLY)
            //   POST /api/chat  — run inference on existing model (READ-ONLY)
            // The following are NEVER called from remote requests:
            //   POST /api/pull    — download model (BLOCKED)
            //   DELETE /api/delete — remove model (BLOCKED)
            //   POST /api/push    — upload model (BLOCKED)
            //   POST /api/create  — create model (BLOCKED)
            //   POST /api/copy    — copy model (BLOCKED)
            const _cOllamaUrl = process.env.OLLAMA_HOST || 'http://localhost:11434';
            let _cModel = '';
            try {
              const _cTags = await fetch(_cOllamaUrl + '/api/tags').then(function(r) { return r.json(); });
              var _cAllModels = (_cTags.models || []).filter(function(m) { return !/embed|nomic/i.test(m.name); });
              // Apply model allowlist — only serve allowed models to remote queries
              var _cModels = _cohereAllowedModels
                ? _cAllModels.filter(function(m) { return _cohereAllowedModels.has(m.name); })
                : _cAllModels;
              if (_cModels.length === 0 && _cohereAllowedModels) {
                dlog('COHERE: no allowed models match downloaded models. Allowlist: ' + [..._cohereAllowedModels].join(', '));
                _cohereStats.queriesErrors++;
                _saveStats();
                continue;
              }
              // Complexity-based model selection (inline version of estimateQueryComplexity)
              var _cQ = _cData.query || '';
              var _cTier = 0; // 0=trivial, 1=moderate, 2=complex
              var _cTokens = Math.ceil(_cQ.length / 4);
              if (_cTokens >= 200) _cTier = 2;
              else if (_cTokens >= 50) _cTier = 1;
              if (/```|functions|classs|imports|defs/.test(_cQ)) _cTier = Math.min(2, _cTier + 1);
              if (/(analyze|compare|explain|describe|differences|step.by.step|comprehensive)/i.test(_cQ)) _cTier = Math.min(2, _cTier + 1);
              var _cGB = 1024 * 1024 * 1024;
              var _cThresh = _cTier === 0 ? 8 * _cGB : _cTier === 1 ? 50 * _cGB : Infinity;
              // Sort ascending by size, pick largest within tier threshold
              _cModels.sort(function(a, b) { return (a.size || 0) - (b.size || 0); });
              // Prefer warm model if it fits
              if (_cLastModel) {
                var _cWarm = _cModels.find(function(m) { return m.name === _cLastModel; });
                if (_cWarm && (_cWarm.size || 0) <= _cThresh) { _cModel = _cWarm.name; }
              }
              if (!_cModel) {
                var _cFit = _cModels.filter(function(m) { return (m.size || 0) <= _cThresh; });
                _cModel = _cFit.length > 0 ? _cFit[_cFit.length - 1].name : (_cModels.length > 0 ? _cModels[_cModels.length - 1].name : '');
              }
              dlog('COHERE routing: tier=' + ['trivial','moderate','complex'][_cTier] + ' model=' + _cModel);
            } catch {}
            if (!_cModel) { dlog('COHERE: no Ollama models available'); _cohereStats.queriesErrors++; _saveStats(); continue; }
            try {
              // SAFETY: Only /api/chat is called — inference on already-downloaded model
              const _cResp = await fetch(_cOllamaUrl + '/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  model: _cModel,
                  messages: [{ role: 'user', content: _cData.query }],
                  stream: false,
                }),
                signal: AbortSignal.timeout(120000),
              });
              const _cResult = await _cResp.json();
              const _cContent = _cResult.message ? _cResult.message.content : '';
              const _cLatency = Date.now() - _cStart;
              // Sign response for accountability (HMAC-SHA256 with peerId)
              const _cSigData = _cData.queryId + ':' + _cContent.slice(0, 200) + ':' + _cModel + ':' + _cLatency;
              const _cSig = createHmac('sha256', nexus.peerId).update(_cSigData).digest('hex').slice(0, 32);
              // Scan inbound query for leaked secrets (defense-in-depth)
              const _cSecretPatterns = [/sk-[a-zA-Z0-9]{20,}/g, /ghp_[a-zA-Z0-9]{36,}/g, /AKIA[0-9A-Z]{16}/g];
              for (const _cPat of _cSecretPatterns) {
                if (_cPat.test(_cData.query)) { dlog('COHERE WARNING: inbound query contains potential secret!'); break; }
              }
              // Publish response
              _natsConn.publish('nexus.cohere.response', _natsCodec.encode(JSON.stringify({
                type: 'cohere.response',
                queryId: _cData.queryId,
                content: _cContent,
                model: _cModel,
                provider: nexus.peerId,
                agentName: nexus.agentName || ('oa-' + hostname().slice(0, 12)),
                latencyMs: _cLatency,
                usage: _cResult.eval_count ? { inputTokens: _cResult.prompt_eval_count || 0, outputTokens: _cResult.eval_count || 0 } : undefined,
                signature: _cSig,
              })));
              _cLastModel = _cModel; // track warm model

              // Stats: track answered
              _cohereStats.queriesAnswered++;
              _cohereStats.totalLatencyMs += _cLatency;
              _cohereStats.bytesOut += (_cContent || '').length;
              _cohereStats.modelsUsed[_cModel] = (_cohereStats.modelsUsed[_cModel] || 0) + 1;
              _saveStats();

              dlog('COHERE response: ' + _cData.queryId + ' model=' + _cModel + ' tier=' + ['trivial','moderate','complex'][_cTier] + ' ' + _cLatency + 'ms');
            } catch (_cErr) {
              dlog('COHERE inference error: ' + (_cErr.message || _cErr));
              _cohereStats.queriesErrors++;
              _saveStats();
            }
          } catch (_cParseErr) {
            // Malformed message — skip
          }
        }
      })().catch(function(e) { dlog('COHERE query handler loop error: ' + (e.message || e)); });
      dlog('COHERE query handler subscribed to nexus.cohere.query');

      // ── WO-1.3: Claim listener — track claims from other nodes ──────
      const _claimSub = _natsConn.subscribe('nexus.cohere.claimed');
      (async function() {
        for await (const _clMsg of _claimSub) {
          try {
            var _clData = JSON.parse(_natsCodec.decode(_clMsg.data));
            if (!_clData.queryId || !_clData.claimedBy) continue;
            // Only store if we haven't claimed it ourselves, or this claim is from someone else
            var _clExisting = cohereClaims.get(_clData.queryId);
            if (!_clExisting) {
              // First claim wins — store it
              cohereClaims.set(_clData.queryId, { claimedBy: _clData.claimedBy, timestamp: _clData.timestamp || Date.now() });
            } else if (_clExisting.claimedBy === nexus.peerId && _clData.claimedBy !== nexus.peerId) {
              // Tie-break: if both claimed, lower peerId wins (deterministic)
              if (_clData.claimedBy < nexus.peerId) {
                cohereClaims.set(_clData.queryId, { claimedBy: _clData.claimedBy, timestamp: _clData.timestamp || Date.now() });
                dlog('COHERE claim tie-break: ' + _clData.queryId + ' → ' + _clData.claimedBy.slice(0, 16) + ' (lower peerId wins)');
              }
            }
            // Prune old claims (60s TTL)
            var _clNow = Date.now();
            for (var [_clK, _clV] of cohereClaims) { if (_clNow - _clV.timestamp > 60000) cohereClaims.delete(_clK); }
          } catch {}
        }
      })().catch(function(e) { dlog('COHERE claim listener error: ' + (e.message || e)); });
      dlog('COHERE claim listener subscribed to nexus.cohere.claimed');

      // ── WO-DL1: Learning ingestion — receive insights from other nodes ──
      const _learnSub = _natsConn.subscribe('nexus.cohere.learning');
      (async function() {
        for await (const _lMsg of _learnSub) {
          if (!cohereActive) continue;
          try {
            var _lData = JSON.parse(_natsCodec.decode(_lMsg.data));
            if (!_lData.insight || !_lData.source_peer) continue;
            // Skip our own insights
            if (_lData.source_peer === nexus.peerId) continue;

            // WO-DL2: Model-tier-aware ingestion filtering
            // Determine local model tier from warm model size
            var _lLocalTier = 'unknown';
            try {
              var _lOllamaUrl = process.env.OLLAMA_HOST || 'http://localhost:11434';
              if (_cLastModel) {
                var _lTagsResp = await fetch(_lOllamaUrl + '/api/tags');
                var _lTags = await _lTagsResp.json();
                var _lModel = (_lTags.models || []).find(function(m) { return m.name === _cLastModel; });
                if (_lModel) {
                  var _lSizeGB = (_lModel.size || 0) / (1024*1024*1024);
                  _lLocalTier = _lSizeGB >= 20 ? 'large' : _lSizeGB >= 5 ? 'medium' : 'small';
                }
              }
            } catch {}

            var _lSourceTier = _lData.model_tier || 'unknown';
            var _lCategory = _lData.category || 'strategy';
            var _lConfidence = _lData.confidence || 0.5;

            // Tier filtering rules:
            // - Same or lower tier → always ingest
            // - Higher tier → only if confidence > 0.8 AND category is recovery (model-agnostic)
            // - Never ingest complex strategies from much larger models to tiny models
            var _lTierOrder = { 'small': 0, 'medium': 1, 'large': 2, 'unknown': 1 };
            var _lLocalLevel = _lTierOrder[_lLocalTier] || 1;
            var _lSourceLevel = _lTierOrder[_lSourceTier] || 1;

            if (_lSourceLevel > _lLocalLevel) {
              // Insight from a LARGER model — apply strict filtering
              if (_lConfidence < 0.8) {
                dlog('COHERE learning SKIPPED (low confidence from higher tier): ' + _lCategory + ' conf=' + _lConfidence);
                continue;
              }
              if (_lCategory !== 'recovery' && _lCategory !== 'debug_heuristic') {
                dlog('COHERE learning SKIPPED (non-recovery from higher tier): ' + _lCategory);
                continue;
              }
            }

            // Ingest: write to local metabolism store
            var _lStoreDir = join(nexusDir, '..', 'memory', 'metabolism');
            var _lStoreFile = join(_lStoreDir, 'store.json');
            var _lStore = [];
            try {
              if (existsSync(_lStoreFile)) _lStore = JSON.parse(readFileSync(_lStoreFile, 'utf8'));
            } catch {}
            // Dedup: skip if we already have this insight (by delta_id)
            if (_lStore.some(function(m) { return m.id === _lData.delta_id; })) continue;
            _lStore.push({
              id: _lData.delta_id,
              type: 'procedural',
              content: '[mesh:' + (_lData.source_agent || 'peer').slice(0, 20) + '] ' + _lData.insight,
              sourceTrace: 'cohere-mesh:' + (_lData.source_peer || '').slice(0, 16),
              scores: {
                novelty: 0.7,
                utility: Math.min(1, _lData.confidence || 0.5),
                confidence: Math.min(1, (_lData.confidence || 0.5) * 0.8), // discount remote slightly
                identityRelevance: 0.2,
              },
              decision: { action: 'admit', reason: 'Ingested from COHERE mesh peer' },
              createdAt: new Date().toISOString(),
              lastAccessedAt: new Date().toISOString(),
              accessCount: 0,
            });
            if (_lStore.length > 100) _lStore = _lStore.slice(-100);
            mkdirSync(_lStoreDir, { recursive: true });
            writeFileSync(_lStoreFile, JSON.stringify(_lStore, null, 2));
            dlog('COHERE learning ingested from ' + (_lData.source_agent || 'unknown') + ': ' + String(_lData.insight).slice(0, 60));

            // WO-DL4: Cross-pin the CID locally if provided (content persistence)
            if (_lData.cid && _lData.cid.startsWith('bafy')) {
              try {
                var _lCidDir = join(nexusDir, 'ipfs', 'cid-registry');
                mkdirSync(_lCidDir, { recursive: true });
                var _lCidFile = join(_lCidDir, 'learning-cids.json');
                var _lCids = {};
                try { if (existsSync(_lCidFile)) _lCids = JSON.parse(readFileSync(_lCidFile, 'utf8')); } catch {}
                _lCids[_lData.delta_id] = { cid: _lData.cid, source: _lData.source_agent, pinned: false, timestamp: Date.now() };
                writeFileSync(_lCidFile, JSON.stringify(_lCids, null, 2));
                dlog('CID registered: ' + _lData.cid.slice(0, 20) + '... from ' + _lData.source_agent);
                // Attempt cross-pin if Helia is ready
                try {
                  if (_heliaReady && _heliaNode) {
                    var { CID: CIDClass } = await import('multiformats/cid');
                    var _lCidObj = CIDClass.parse(_lData.cid);
                    for await (var _lPin of _heliaNode.pins.add(_lCidObj)) {}
                    _lCids[_lData.delta_id].pinned = true;
                    writeFileSync(_lCidFile, JSON.stringify(_lCids, null, 2));
                    dlog('Cross-pinned CID: ' + _lData.cid.slice(0, 20));
                  }
                } catch {}
              } catch {}
            }
          } catch {}
        }
      })().catch(function(e) { dlog('COHERE learning listener error: ' + (e.message || e)); });
      dlog('COHERE learning listener subscribed to nexus.cohere.learning');

      // ── WO-1.5: Capacity announcement ───────────────────────────────
      // Publish what models we have, system metrics, and warm model status
      // so other nodes can make intelligent routing decisions.
      var _capAnnounceInterval = null;
      async function _publishCapacityAnnouncement() {
        if (!cohereActive || !_natsConn || !_natsCodec) return;
        try {
          var _capOllamaUrl = process.env.OLLAMA_HOST || 'http://localhost:11434';
          var _capModels = [];
          try {
            var _capTags = await fetch(_capOllamaUrl + '/api/tags').then(function(r) { return r.json(); });
            _capModels = (_capTags.models || []).map(function(m) {
              return {
                name: m.name,
                size: m.size || 0,
                family: m.details ? m.details.family || '' : '',
                parameterSize: m.details ? m.details.parameter_size || '' : '',
                quantization: m.details ? m.details.quantization_level || '' : '',
              };
            });
          } catch {}
          // Filter by allowlist
          if (_cohereAllowedModels) {
            _capModels = _capModels.filter(function(m) { return _cohereAllowedModels.has(m.name); });
          }
          var _capMetrics = await _collectSysMetrics();
          var announcement = {
            type: 'capacity.announcement',
            peerId: nexus.peerId,
            agentName: agentName,
            agentType: agentType,
            cohereActive: cohereActive,
            models: _capModels,
            warmModel: _cLastModel || null,
            modelCount: _capModels.length,
            systemMetrics: _capMetrics,
            allowedModels: _cohereAllowedModels ? [..._cohereAllowedModels] : null,
            stats: {
              queriesAnswered: _cohereStats.queriesAnswered,
              avgLatencyMs: _cohereStats.queriesAnswered > 0 ? Math.round(_cohereStats.totalLatencyMs / _cohereStats.queriesAnswered) : 0,
            },
            timestamp: Date.now(),
          };
          _natsConn.publish('nexus.agents.capacity', _natsCodec.encode(JSON.stringify(announcement)));

          // Re-announce sponsorship periodically so consumers can discover us
          // (NATS pub/sub is ephemeral — consumers only see messages while subscribed)
          if (globalThis._activeSponsorData && _natsConn) {
            var _spData = globalThis._activeSponsorData;
            _spData.timestamp = Date.now(); // refresh timestamp
            _natsConn.publish('nexus.sponsors.announce', _natsCodec.encode(JSON.stringify(_spData)));
            // Also re-send to sponsors room
            if (rooms.has('sponsors')) {
              try { rooms.get('sponsors').send(JSON.stringify(_spData), { format: 'application/json' }); } catch {}
            }
          }

          // Also publish enriched discovery announcement so the dashboard sees
          // IPFS/memory/identity/emotional state in sidebar cards (WO-VIS1 fields)
          try {
            // Gather memory metrics
            var _memCount = 0;
            var _memSentiment = 'neutral';
            var _ipfsBytes = 0;
            try {
              var _metaFile = join(nexusDir, '..', 'memory', 'metabolism', 'store.json');
              if (existsSync(_metaFile)) {
                var _mStore = JSON.parse(readFileSync(_metaFile, 'utf8'));
                _memCount = _mStore.filter(function(m) { return m.type !== 'quarantine'; }).length;
                var _recov = _mStore.filter(function(m) { return m.content && m.content.startsWith('[recovery]'); }).length;
                var _strat = _mStore.filter(function(m) { return m.content && m.content.startsWith('[strategy]'); }).length;
                _memSentiment = _strat > _recov ? 'proactive' : _recov > 0 ? 'defensive' : 'neutral';
              }
            } catch {}
            try {
              var _blocksDir = join(nexusDir, 'ipfs', 'blocks');
              if (existsSync(_blocksDir)) {
                var _walkBytes = function(d) { var t = 0; try { var ent = readdirSync(d, {withFileTypes:true}); for (var i=0;i<ent.length;i++) { if (ent[i].isDirectory()) t += _walkBytes(join(d,ent[i].name)); else try { t += statSync(join(d,ent[i].name)).size; } catch {} } } catch {} return t; };
                _ipfsBytes = _walkBytes(_blocksDir);
              }
            } catch {}
            // Identity CID
            var _idCid = '';
            try {
              var _cidFile = join(nexusDir, '..', 'identity', 'cids.json');
              if (existsSync(_cidFile)) { var _cids = JSON.parse(readFileSync(_cidFile, 'utf8')); _idCid = _cids.latest || ''; }
            } catch {}

            var discoveryAnn = {
              type: 'nexus.announce',
              peerId: nexus.peerId,
              agentName: agentName,
              rooms: [],
              multiaddrs: [],
              timestamp: Date.now(),
              capabilities: _capModels.map(function(m) { return m.name; }),
              identityCid: _idCid || undefined,
              identityCoherence: 0.9,
              memoryCount: _memCount,
              memorySentiment: _memSentiment,
              ipfsStorageBytes: _ipfsBytes,
              emotionalState: cohereActive ? 'focused' : 'neutral',
              taskRate: (_cohereStats.queriesAnswered || 0) / Math.max(1, (Date.now() - (_cohereStats._startTime || Date.now())) / 3600000),
              cohereLearnings: _cohereStats.queriesSent || 0,
            };
            _natsConn.publish('nexus.agents.discovery', _natsCodec.encode(JSON.stringify(discoveryAnn)));
          } catch (e) { dlog('Discovery announcement error: ' + (e.message || e)); }

          dlog('Capacity announcement published: ' + _capModels.length + ' models, warm=' + (_cLastModel || 'none'));
        } catch (e) {
          dlog('Capacity announcement error: ' + (e.message || e));
        }
      }
      // Publish immediately on COHERE activation
      if (cohereActive) _publishCapacityAnnouncement();
      // Re-announce every 60s while COHERE is active
      _capAnnounceInterval = setInterval(function() {
        if (cohereActive) _publishCapacityAnnouncement();
      }, 60000);

      // ── WO-DL3: Epoch sync — hash-based state comparison every 5 min ──
      // Each node publishes a lightweight fingerprint of its memory state.
      // If hashes differ between same-tier nodes, the lagging node can
      // request missing insights via nexus.cohere.learning.request.
      var _epochCounter = 0;
      function _publishEpochSync() {
        if (!cohereActive || !_natsConn || !_natsCodec) return;
        try {
          // Read local memory store for fingerprinting
          var _esStoreFile = join(nexusDir, '..', 'memory', 'metabolism', 'store.json');
          var _esStore = [];
          try { if (existsSync(_esStoreFile)) _esStore = JSON.parse(readFileSync(_esStoreFile, 'utf8')); } catch {}

          // Compute top-10 memory IDs sorted by utility*confidence (deterministic)
          var _esFiltered = _esStore
            .filter(function(m) { return m.type !== 'quarantine' && m.scores && m.scores.confidence > 0.15; })
            .sort(function(a, b) { return (b.scores.utility * b.scores.confidence) - (a.scores.utility * a.scores.confidence); })
            .slice(0, 10);
          var _esTopIds = _esFiltered.map(function(m) { return m.id; }).sort().join(',');

          // SHA-256 hash of top-10 IDs for compact comparison
          var _esHash = createHash('sha256').update(_esTopIds).digest('hex').slice(0, 16);

          _epochCounter++;
          var _esAnnouncement = {
            type: 'cohere.epoch',
            peer: nexus.peerId,
            agentName: agentName,
            epoch: _epochCounter,
            memoryCount: _esStore.length,
            topHash: _esHash,
            insightsAvailable: _esFiltered.length,
            timestamp: Date.now(),
          };
          _natsConn.publish('nexus.cohere.learning.epoch', _natsCodec.encode(JSON.stringify(_esAnnouncement)));
          dlog('Epoch sync published: epoch=' + _epochCounter + ' memories=' + _esStore.length + ' hash=' + _esHash);
        } catch (e) {
          dlog('Epoch sync error: ' + (e.message || e));
        }
      }

      // Epoch sync every 5 minutes
      setInterval(function() {
        if (cohereActive) _publishEpochSync();
      }, 300000);
      // Also publish on startup after a brief delay
      setTimeout(function() { if (cohereActive) _publishEpochSync(); }, 10000);

      // Listen for epoch announcements from other nodes
      var _epochSub = _natsConn.subscribe('nexus.cohere.learning.epoch');
      (async function() {
        for await (var _eMsg of _epochSub) {
          try {
            var _eData = JSON.parse(_natsCodec.decode(_eMsg.data));
            if (!_eData.peer || _eData.peer === nexus.peerId) continue;
            // Log peer epoch state for dashboard visibility
            dlog('Epoch from ' + (_eData.agentName || 'peer') + ': epoch=' + _eData.epoch + ' memories=' + _eData.memoryCount + ' hash=' + _eData.topHash);
            // Future: if our hash differs and they have more insights, request sync
          } catch {}
        }
      })().catch(function(e) { dlog('Epoch listener error: ' + (e.message || e)); });
      dlog('Epoch sync listener active on nexus.cohere.learning.epoch');

      // Lazy-init IPFS in background (non-blocking)
      _ensureHelia().catch(function() {});
    }

    writeStatus();
    console.log('Nexus daemon connected as ' + nexus.peerId);
  } catch (err) {
    writeStatus({ error: err.message || String(err) });
    console.error('Nexus daemon connect failed:', err.message || err);
    process.exit(1);
  }
})();

// Ignore SIGPIPE — broken pipe from network writes should NOT kill daemon
try { process.on('SIGPIPE', function() { dlog('SIGPIPE received — ignored'); }); } catch {}

// Graceful shutdown
process.on('SIGTERM', async () => {
  for (const [, room] of rooms) { try { await room.leave(); } catch {} }
  try { await nexus.disconnect(); } catch {}
  try { unlinkSync(pidFile); } catch {}
  try { unlinkSync(statusFile); } catch {}
  process.exit(0);
});
process.on('SIGINT', () => process.emit('SIGTERM'));
