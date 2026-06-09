"""Public placeholder branch file for fixture episodes."""


class StructuredRoutingBranch:
    """Claimed structured routing branch."""

    def __call__(self, attention_output, structured_routing_output, alpha=1.0):
        return attention_output + alpha * structured_routing_output

