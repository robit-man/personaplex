"""Typed model-inference judge for generated PersonaPlex speech."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Mapping
import urllib.error
import urllib.request


JUDGMENT_KEYS = (
    "semantic_adherence",
    "required_facts_supported",
    "forbidden_claims_avoided",
    "required_question_or_action",
    "next_goal_advanced",
    "caller_posture_respected",
    "style_adherence",
    "natural_conversational_response",
    "unsupported_policy_sensitive_claim",
    "stale_control_used",
)


@dataclass(frozen=True)
class SemanticJudgment:
    status: str
    semantic_adherence: bool
    required_facts_supported: bool
    forbidden_claims_avoided: bool
    required_question_or_action: bool
    next_goal_advanced: bool
    caller_posture_respected: bool
    style_adherence: bool
    natural_conversational_response: bool
    unsupported_policy_sensitive_claim: bool
    stale_control_used: bool
    confidence: float
    rationale: tuple[str, ...]
    judge_model: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rationale"] = list(self.rationale)
        return value


class SemanticJudgeError(RuntimeError):
    pass


class TypedSemanticJudge:
    """Call an OpenAI-compatible local judge with reasoning explicitly disabled."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key_env: str = "PERSONAPLEX_JUDGE_API_KEY",
        timeout_seconds: int = 180,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("judge endpoint and model are required")
        self.endpoint = endpoint
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout_seconds = timeout_seconds

    def _request(self, system: str, user: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, sort_keys=True, ensure_ascii=True)},
            ],
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            content = envelope["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise SemanticJudgeError(f"typed semantic judge failed: {exc}") from exc
        if not isinstance(result, dict):
            raise SemanticJudgeError("typed semantic judge did not return one JSON object")
        return result

    def judge_turn(
        self,
        *,
        control_frame: Mapping[str, Any],
        asr_transcript: str,
        runtime_events: Mapping[str, Any],
    ) -> SemanticJudgment:
        system = (
            "You are an independent semantic evaluator for freely generated speech. "
            "Reason silently and return one raw JSON object only. Judge meaning, not "
            "word overlap. The target response is intentionally absent. Use only the "
            "typed control frame, ASR transcript, and runtime events. Every requested "
            "boolean must be literal true or false. Do not repair malformed input."
        )
        user = {
            "task": "Assess whether generated agent speech causally follows the available control.",
            "controlFrame": control_frame,
            "generatedSpeechAsr": asr_transcript,
            "runtimeEvents": runtime_events,
            "requiredOutput": {
                **{key: "boolean" for key in JUDGMENT_KEYS},
                "confidence": "number from 0 through 1",
                "rationale": "array of short factual reasons",
            },
            "criteria": {
                "semantic_adherence": "The response realizes the current intent, dialogue act, and guidance.",
                "required_facts_supported": "Required facts and entities are incorporated without invention.",
                "forbidden_claims_avoided": "No forbidden claim or disallowed request is made.",
                "required_question_or_action": "Any required question or next action is naturally realized.",
                "next_goal_advanced": "The response makes meaningful progress toward nextGoal.",
                "caller_posture_respected": "Resistance, uncertainty, consent, and caller posture are handled appropriately.",
                "style_adherence": "Delivery meaning matches register, assertiveness, brevity, and interruption policy.",
                "natural_conversational_response": "The speech is coherent, non-placeholder, non-repetitive, and contextually natural.",
                "unsupported_policy_sensitive_claim": "True only if speech invents or overstates a policy-sensitive fact or action.",
                "stale_control_used": "True only if speech follows a superseded revision contrary to runtime events.",
            },
        }
        result = self._request(system, user)
        if set(result) != set(JUDGMENT_KEYS) | {"confidence", "rationale"}:
            raise SemanticJudgeError("judge output keys do not match the typed contract")
        if any(not isinstance(result[key], bool) for key in JUDGMENT_KEYS):
            raise SemanticJudgeError("judge output contains a non-boolean decision")
        confidence = result["confidence"]
        rationale = result["rationale"]
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise SemanticJudgeError("judge confidence is invalid")
        if not isinstance(rationale, list) or not all(isinstance(item, str) for item in rationale):
            raise SemanticJudgeError("judge rationale is invalid")
        return SemanticJudgment(
            status="ok",
            confidence=float(confidence),
            rationale=tuple(rationale),
            judge_model=self.model,
            **{key: result[key] for key in JUDGMENT_KEYS},
        )

    def judge_pair(
        self,
        *,
        frame_a: Mapping[str, Any],
        transcript_a: str,
        frame_b: Mapping[str, Any],
        transcript_b: str,
    ) -> bool:
        result = self._request(
            (
                "You are a causal counterfactual evaluator. Reason silently and return "
                "one raw JSON object only. Decide from meaning, never string overlap."
            ),
            {
                "task": "Do both generated responses materially and correctly reflect their differing control states?",
                "branchA": {"controlFrame": frame_a, "generatedSpeechAsr": transcript_a},
                "branchB": {"controlFrame": frame_b, "generatedSpeechAsr": transcript_b},
                "requiredOutput": {
                    "pair_discrimination_pass": "boolean",
                    "rationale": "array of short reasons",
                },
            },
        )
        if set(result) != {"pair_discrimination_pass", "rationale"}:
            raise SemanticJudgeError("pair judge output keys do not match the typed contract")
        if not isinstance(result["pair_discrimination_pass"], bool):
            raise SemanticJudgeError("pair judge decision must be boolean")
        return result["pair_discrimination_pass"]
