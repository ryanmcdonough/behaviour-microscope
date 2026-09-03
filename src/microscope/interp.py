"""The only file that talks to Neuronpedia's interp-engine.

interp-engine is the interpretability runtime for this project. Nothing here implements
hooks, activation capture, model architecture mappings or steering -- those are the engine's
job, and this module is a thin adapter that speaks the experiment's vocabulary (a prompt, a
layer, a patch) to the engine's (token ids, an ``Address``, a ``SteeringSpec``).

Two engine behaviours the rest of the codebase depends on, both documented:

* ``steer()`` is a context manager and every forward inside it is steered. Its ``position_mask``
  names prompt positions to **exclude**, so patching one position means masking every other.
* ``AddSpec`` is applied as ``coeff * vector`` (``interp_engine.steer.steer_delta``). Replacing
  an activation is therefore expressible as adding the difference between the two activations,
  which is what ``patched_next_token_logits`` does.

See RESEARCH.md for why activation patching is built out of an additive steer rather than a
bespoke write hook.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from interp_engine import (
    AddSpec,
    LayerSteeringSpec,
    SteeringSpec,
    generate_stream,
    load_model,
    run_with_cache,
    steer,
    sync_model,
)

RESIDUAL_POINT = "resid_post"

# First-token spellings of an answer letter. Summed rather than maxed: the quantity we want is
# the model's probability of answering with that letter, however it happens to tokenize.
LETTER_VARIANTS = ("{}", " {}", "{}.", " {}.", "{})", "\n{}")


@dataclass
class ModelHandle:
    """A loaded model plus the small amount of derived state the experiment reuses."""

    model: object
    sync: object
    model_id: str
    backend: str
    n_layers: int
    d_model: int
    letter_token_ids: dict[str, list[int]]
    # Only consulted on models whose chat template reads it. False keeps the answer in the
    # first generated token, which is what the forced-choice measurement requires.
    enable_thinking: bool = False

    @property
    def last_layer(self) -> int:
        return self.n_layers - 1

    @property
    def final_point(self) -> str:
        return f"{RESIDUAL_POINT}.{self.last_layer}"

    def layer_points(self) -> list[str]:
        return [f"{RESIDUAL_POINT}.{layer}" for layer in range(self.n_layers)]

    def shutdown(self) -> None:
        self.sync.shutdown()


def open_model(
    model_id: str,
    *,
    backend: str = "eager",
    dtype: str | None = None,
    enable_thinking: bool = False,
    **kwargs,
) -> ModelHandle:
    """Load a model through interp-engine and warm it up.

    ``backend="eager"`` is the default here rather than the engine's ``"auto"`` ladder. See
    RESEARCH.md: the vLLM backend must be built with ``enforce_eager=True`` to capture at all,
    which removes most of its speed advantage at this model size, and the eager backend is the
    one that serves every point and fills in ``GenStep.logits`` for the cross-check in
    ``verify_logit_path``.

    That choice has a sharp edge: ``load_model``'s own CUDA auto-detection only runs for
    ``backend="auto"``. Forcing ``"eager"`` here bypasses that ladder entirely, and
    ``EagerModel`` with no ``device``/``device_map`` falls through to whatever
    ``transformers.AutoModelForCausalLM.from_pretrained`` does with neither -- CPU, silently,
    even with a GPU sitting idle. So this fills the same default back in, unless the caller
    named a device explicitly (a caller that passed ``device="cpu"`` for a small local test,
    say, must not be overridden).
    """
    if dtype is not None:
        kwargs["dtype"] = dtype
    if backend == "eager" and "device" not in kwargs and "device_map" not in kwargs:
        kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_id, backend=backend, **kwargs)
    sync = sync_model(model)
    sync.warmup()
    handle = ModelHandle(
        model=model,
        sync=sync,
        model_id=model_id,
        backend=type(model).__name__,
        n_layers=model.n_layers,
        d_model=model.d_model,
        letter_token_ids={},
        enable_thinking=enable_thinking,
    )
    handle.letter_token_ids = _answer_token_ids(model)
    return handle


def _answer_token_ids(model) -> dict[str, list[int]]:
    """First-token ids for each answer letter, with any id both letters share removed.

    A variant whose first token is punctuation rather than the letter -- ``"**A"`` tokenizes to
    the ``**`` token on several vocabularies -- lands the same id in both sets, and summing over
    it would count the same probability mass for A and for B. Shared ids carry no information
    about which letter is coming, so they are dropped rather than assigned to one side.
    """
    per_letter = {letter: _letter_token_ids(model, letter) for letter in ("A", "B")}
    shared = set(per_letter["A"]) & set(per_letter["B"])
    resolved = {letter: [i for i in ids if i not in shared] for letter, ids in per_letter.items()}
    for letter, ids in resolved.items():
        if not ids:
            raise RuntimeError(
                f"No token id uniquely starts the answer {letter!r} on this tokenizer; "
                "the forced-choice measurement cannot be scored."
            )
    return resolved


def _letter_token_ids(model, letter: str) -> list[int]:
    """Distinct first-token ids for the ways this tokenizer might start the answer ``letter``.

    ``prepend_bos=False`` matters: with the BOS token attached, every variant's first token is
    the BOS token and the whole set collapses to one meaningless id.
    """
    ids: list[int] = []
    for template in LETTER_VARIANTS:
        encoded = model.tok.to_tokens(template.format(letter), prepend_bos=False)[0].tolist()
        if encoded and encoded[0] not in ids:
            ids.append(encoded[0])
    if not ids:
        raise RuntimeError(f"Could not tokenize the answer letter {letter!r}")
    return ids


def tokenize_prompt(handle: ModelHandle, prompt: str) -> list[int]:
    """Token ids for a prompt, through the model's own chat template where it has one.

    Hand-writing a chat format is the one thing the engine refuses to do for you, and for good
    reason, so this asks the tokenizer and falls back to plain completion only when the model
    genuinely has no chat format.
    """
    model = handle.model
    if model.tok.has_chat_template():
        # A hybrid-reasoning model (Qwen3, and others) leaves the assistant turn open by
        # default, so its first generated token is <think> rather than the answer. The
        # forced-choice measurement reads the first token, so that would put essentially no
        # probability on either letter and quietly turn the run into noise -- the letter_mass
        # quality check would flag it, but only after the run.
        #
        # enable_thinking=False makes the template close an empty reasoning block itself, so
        # the next token is the answer. Asked rather than assumed: a renderer ignores or
        # rejects a kwarg it does not know, so the engine reports which ones it reads.
        # Reasoning-vs-not is a variable worth studying (see RESEARCH.md); this pins it to a
        # known state rather than leaving it to each model's default.
        template_kwargs: dict = {}
        if "enable_thinking" in model.tok.accepted_template_kwargs(["enable_thinking"]):
            template_kwargs["enable_thinking"] = handle.enable_thinking
        return list(
            model.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, **template_kwargs
            )
        )
    # No chat format: complete the prompt directly rather than inventing a template the model
    # was never trained on, which is what the engine refuses to do for you.
    return model.tok.to_tokens(prompt + "\n\nAnswer:")[0].tolist()


def _rows(tensor: torch.Tensor) -> torch.Tensor:
    """Drop a leading batch dimension of 1, so every caller sees ``[n_tokens, width]``."""
    return tensor[0] if tensor.ndim == 3 else tensor


def _ids_on_device(handle: ModelHandle, token_ids: list[int]) -> torch.Tensor:
    """Token ids on the same device as the weights, which ``run_with_cache`` does not do itself.

    The engine's ``generate_stream`` already places ids on ``model.device``. Its ``run_with_cache``
    free function does not: a Python list becomes a CPU tensor via ``as_batched_tokens``, and the
    embedding lookup then fails with a device mismatch once the weights are on CUDA. The engine's
    own ``EagerModel.capture`` works around the same gap by building the tensor on ``self.device``
    before calling that free function; this is that workaround for the path the rest of this file
    uses. See RESEARCH.md.
    """
    return torch.tensor(token_ids, dtype=torch.long, device=handle.model.device)


def capture_residuals(handle: ModelHandle, token_ids: list[int]) -> dict[int, torch.Tensor]:
    """Residual stream at the final prompt position, for every layer.

    Only the final position is kept. It is the position the next-token distribution is read
    from, and it is the one position that is unambiguously matched across two prompts of
    different token length -- see RESEARCH.md on the alignment problem.
    """
    cache = run_with_cache(handle.model, _ids_on_device(handle, token_ids), handle.layer_points())
    return {
        layer: _rows(cache[f"{RESIDUAL_POINT}.{layer}"])[-1].detach().float().cpu()
        for layer in range(handle.n_layers)
    }


def next_token_logits(handle: ModelHandle, token_ids: list[int]) -> torch.Tensor:
    """Full next-token logit vector, via the engine's own unembed.

    Reads the last layer's ``resid_post`` at the final prompt position and sends it through
    ``decode_residuals``, which applies the model's configured ``final_logit_softcapping`` --
    required for Gemma-2, and the reason this uses the model *method* rather than the free
    function. ``verify_logit_path`` checks the result against the sampler's own logits.
    """
    cache = run_with_cache(handle.model, _ids_on_device(handle, token_ids), [handle.final_point])
    residual = _rows(cache[handle.final_point])[-1]
    return handle.sync.decode_residuals(residual.unsqueeze(0))[0].detach().float().cpu()


def patched_next_token_logits(
    handle: ModelHandle,
    token_ids: list[int],
    layer: int,
    delta: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Next-token logits after adding ``scale * delta`` to ``resid_post.layer``, final position only.

    With ``delta = donor_activation - recipient_activation`` and ``scale=1.0`` this is an exact
    activation patch: the recipient's activation at that layer and position is replaced by the
    donor's. Nothing upstream of ``layer`` is modified, so the value the hook sees is the one
    the delta was measured against.

    The vector is passed pre-normalized with the norm carried in ``scale`` because the engine's
    ``AddSpec`` docstring specifies a unit vector, and both backends then apply the same
    arithmetic whether or not they normalize again.
    """
    norm = float(torch.linalg.vector_norm(delta.float()))
    if norm == 0.0:
        raise ValueError("Cannot build a patch from a zero delta; use scale=0.0 for a no-op control")
    spec = SteeringSpec(
        layers={layer: LayerSteeringSpec(operations=[AddSpec(vector=delta.float() / norm, scale=norm * scale)])},
        point=RESIDUAL_POINT,
    )
    # Everything except the final position is excluded, which is what makes this a single-position
    # patch rather than a steer applied across the whole prompt.
    mask = list(range(len(token_ids) - 1))
    ids = _ids_on_device(handle, token_ids)
    with steer(handle.model, spec, prompt_token_ids=ids, position_mask=mask):
        cache = run_with_cache(handle.model, ids, [handle.final_point])
        residual = _rows(cache[handle.final_point])[-1]
    return handle.sync.decode_residuals(residual.unsqueeze(0))[0].detach().float().cpu()


def letter_probabilities(handle: ModelHandle, logits: torch.Tensor) -> dict[str, float]:
    """P(A) and P(B) over the full vocabulary, plus their renormalized forced-choice split."""
    probs = torch.softmax(logits, dim=-1)
    mass = {letter: float(probs[ids].sum()) for letter, ids in handle.letter_token_ids.items()}
    total = mass["A"] + mass["B"]
    return {
        "p_a": mass["A"],
        "p_b": mass["B"],
        "letter_mass": total,
        "p_a_norm": mass["A"] / total if total > 0 else 0.5,
        "p_b_norm": mass["B"] / total if total > 0 else 0.5,
    }


def generate_answer(handle: ModelHandle, token_ids: list[int], max_tokens: int = 24) -> str:
    """Greedy continuation, kept as the qualitative record of what the model actually said."""
    return "".join(
        step.token_str for step in generate_stream(handle.model, token_ids, max_tokens=max_tokens, temperature=0.0)
    )


def verify_logit_path(handle: ModelHandle, token_ids: list[int], tolerance: float = 5e-3) -> dict:
    """Check the decode_residuals logits against the sampler's own, where the backend has them.

    ``GenStep.logits`` is filled in on the eager backend and ``None`` on vLLM. Where it is
    available this confirms that reading the last layer's residual through ``decode_residuals``
    reproduces the model's real next-token distribution, rather than a logit-lens approximation
    of it. Returns ``{"checked": False}`` where the backend cannot answer.
    """
    decoded = next_token_logits(handle, token_ids)
    sampler = None
    for step in generate_stream(handle.model, token_ids, max_tokens=1, temperature=0.0):
        sampler = step.logits
        break
    if sampler is None:
        return {"checked": False, "reason": f"{handle.backend} does not return GenStep.logits"}
    sampler = sampler.reshape(-1, sampler.shape[-1])[-1].detach().float().cpu()
    a = torch.softmax(decoded, dim=-1)
    b = torch.softmax(sampler, dim=-1)
    max_abs = float((a - b).abs().max())
    return {
        "checked": True,
        "max_abs_prob_difference": max_abs,
        "argmax_agrees": bool(decoded.argmax() == sampler.argmax()),
        "within_tolerance": max_abs <= tolerance,
    }
