
def normalize(input, p=2, dim=1, eps=1e-12):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.
    Does:
    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}
    for each subtensor v over dimension dim of input. Each subtensor is flattened into a vector,
    i.e. :math:`\lVert v \rVert_p` is not a matrix norm.
    With default arguments normalizes over the second dimension with Euclidean norm.
    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation
        dim (int): the dimension to reduce
        eps (float): small value to avoid division by zero
    """
    return input / input.norm(p, dim).clamp(min=eps).expand_as(input)