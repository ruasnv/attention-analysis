class BaseMethod(ABC):

    # --- Setup ---
    @abstractmethod
    def build(self, model_name: str) -> None:
        """Load pretrained model, apply weight surgery if needed, configure parameters."""

    # --- Forward ---
    @abstractmethod
    def forward(self, input_ids, attention_mask, labels) -> loss
        """Single forward pass returning scalar loss."""

    # --- Optimizer ---
    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Return configured optimizer. Default: AdamW on trainable params.
        PoLAR overrides this to return a landing-field-aware optimizer."""

    def pre_optimizer_step(self) -> None:
        """Hook called after loss.backward(), before optimizer.step().
        Default: no-op. PoLAR overrides to apply landing field gradient modification."""

    # --- Introspection ---
    def trainable_parameters(self) -> Iterator[nn.Parameter]
    def num_trainable_params(self) -> int
    def parameter_summary(self) -> dict   # name → requires_grad, shape, count

    # --- Checkpointing ---
    def state_dict(self) -> dict          # everything needed to resume training
    def load_state_dict(self, d: dict) -> None

    # --- Analysis (called every epoch, all methods) ---
    def geometric_health_snapshot(self) -> dict
        """Compute stable_rank, sv_entropy, effective_rank, condition_number,
        nuclear_norm, isotropy on W_V and W_O per head. Works by calling
        get_live_WV_WO() which each method implements."""

    @abstractmethod
    def get_live_WV_WO(self) -> dict[str, Tensor]
        """Return the actual live weight matrices (post-reconstruction for PAFT/SVF,
        W_0 + ΔW for additive methods, W_0 for frozen). Used by geometric_health_snapshot."""

    # --- PAFT-specific (returns None for baselines) ---
    def paft_snapshot(self) -> dict | None:
        """Return Q, S, lambda, EV tensors for PAFT methods. None for all others."""
        return None