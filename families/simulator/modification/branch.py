"""Public structured branch sketch."""


class StructuredRoutingBranch:
    """Claimed branch: align queries with relevant key-value routing structure."""

    def combine(self, attention_output, structured_routing_output, alpha=1.0):
        return attention_output + alpha * structured_routing_output

