# PersonaPlex local runtime topology contract

`/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env` is the single source of truth for local PersonaPlex ports, host binding, semantic model, context window, and physical GPU assignments. Its versioned template is `config/personaplex-runtime.env.example`.

The control service, ChatML proxy, Chatterbox worker, synthesis lanes, and certifiers source this contract. They must not introduce local endpoint literals or fall back to another Ollama listener.

The intended chains are:

`synthesis lane or certifier -> ChatML proxy -> bounded Ornith controller`

`synthesis lane -> shared Chatterbox worker`

After changing the contract, restart the controller, proxy, renderer, lanes, and certifiers in that order. Run `tools/personaplex-runtime-status.sh` before resuming work. It emits the resolved endpoints, proxy upstream/model/context, Voicebox health, and control-model residency. Missing or stale components fail before lanes or certifiers can produce records.
