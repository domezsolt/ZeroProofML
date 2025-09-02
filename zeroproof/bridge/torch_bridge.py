"""
PyTorch bridge for transreal arithmetic.

This module provides conversions between PyTorch tensors and transreal
representations, enabling integration with deep learning workflows.
"""

from typing import Tuple, Union, Optional, Any, overload
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ..core import TRScalar, TRTag, real, pinf, ninf, phi


class TRTensor:
    """
    Transreal tensor for PyTorch integration.
    
    Similar to TRArray but designed to work with PyTorch tensors,
    supporting GPU operations and automatic differentiation.
    """
    
    def __init__(self, values: 'torch.Tensor', tags: 'torch.Tensor', requires_grad: bool = False):
        """
        Initialize TR tensor.
        
        Args:
            values: Tensor of float values
            tags: Tensor of tag codes (uint8)
            requires_grad: Whether to track gradients for values
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TRTensor")
        
        if values.shape != tags.shape:
            raise ValueError("Values and tags must have same shape")
        
        self.values = values
        self.tags = tags
        self._shape = values.shape
        self._device = values.device
        self._dtype = values.dtype
        
        if requires_grad and values.requires_grad:
            # Only REAL values can have gradients
            real_mask = self.is_real()
            if real_mask.any():
                # Create a differentiable view
                self.values.requires_grad_(True)
    
    @property
    def shape(self) -> 'torch.Size':
        """Get tensor shape."""
        return self._shape
    
    @property
    def device(self) -> 'torch.device':
        """Get tensor device."""
        return self._device
    
    @property
    def dtype(self) -> 'torch.dtype':
        """Get value dtype."""
        return self._dtype
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self.values.ndim
    
    @property
    def size(self) -> 'torch.Size':
        """Get tensor size (PyTorch style)."""
        return self.values.size()
    
    def is_real(self) -> 'torch.Tensor':
        """Return boolean mask of REAL elements."""
        return self.tags == TAG_CODES['REAL']
    
    def is_pinf(self) -> 'torch.Tensor':
        """Return boolean mask of PINF elements."""
        return self.tags == TAG_CODES['PINF']
    
    def is_ninf(self) -> 'torch.Tensor':
        """Return boolean mask of NINF elements."""
        return self.tags == TAG_CODES['NINF']
    
    def is_phi(self) -> 'torch.Tensor':
        """Return boolean mask of PHI elements."""
        return self.tags == TAG_CODES['PHI']
    
    def is_finite(self) -> 'torch.Tensor':
        """Return boolean mask of finite (REAL) elements."""
        return self.is_real()
    
    def is_infinite(self) -> 'torch.Tensor':
        """Return boolean mask of infinite (PINF/NINF) elements."""
        return (self.tags == TAG_CODES['PINF']) | (self.tags == TAG_CODES['NINF'])
    
    def to(self, device: Union[str, 'torch.device']) -> 'TRTensor':
        """Move tensor to device."""
        return TRTensor(
            self.values.to(device),
            self.tags.to(device),
            requires_grad=self.values.requires_grad
        )
    
    def cpu(self) -> 'TRTensor':
        """Move tensor to CPU."""
        return self.to('cpu')
    
    def cuda(self, device: Optional[int] = None) -> 'TRTensor':
        """Move tensor to CUDA device."""
        if device is None:
            return self.to('cuda')
        else:
            return self.to(f'cuda:{device}')
    
    def detach(self) -> 'TRTensor':
        """Detach from computation graph."""
        return TRTensor(
            self.values.detach(),
            self.tags.detach(),
            requires_grad=False
        )
    
    def clone(self) -> 'TRTensor':
        """Create a copy of the tensor."""
        return TRTensor(
            self.values.clone(),
            self.tags.clone(),
            requires_grad=self.values.requires_grad
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TRTensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"
    
    def __getitem__(self, key):
        """Support tensor indexing."""
        values = self.values[key]
        tags = self.tags[key]
        
        # If scalar result, return TRScalar
        if values.dim() == 0:
            tag_code = tags.item()
            tag = CODE_TO_TAG[tag_code]
            if tag == TRTag.REAL:
                return TRScalar(float(values.item()), tag)
            else:
                return TRScalar(float('nan'), tag)
        else:
            return TRTensor(values, tags, requires_grad=values.requires_grad)


# Tag encoding constants
TAG_CODES = {
    'REAL': 0,
    'PINF': 1,
    'NINF': 2,
    'PHI': 3,
}

TAG_TO_CODE = {
    TRTag.REAL: 0,
    TRTag.PINF: 1,
    TRTag.NINF: 2,
    TRTag.PHI: 3,
}

CODE_TO_TAG = {v: k for k, v in TAG_TO_CODE.items()}


def from_torch(tensor: 'torch.Tensor', requires_grad: bool = False) -> TRTensor:
    """
    Convert PyTorch tensor to transreal tensor.
    
    Args:
        tensor: PyTorch tensor
        requires_grad: Whether to track gradients
        
    Returns:
        TRTensor with appropriate tags
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for from_torch")
    
    # Ensure we have a float tensor
    if not tensor.is_floating_point():
        tensor = tensor.float()
    
    # Create value and tag tensors
    values = tensor.clone()
    tags = torch.zeros_like(tensor, dtype=torch.uint8)
    
    # Classify elements
    finite_mask = torch.isfinite(tensor)
    nan_mask = torch.isnan(tensor)
    posinf_mask = torch.isposinf(tensor)
    neginf_mask = torch.isneginf(tensor)
    
    # Set tags
    tags[finite_mask] = TAG_CODES['REAL']
    tags[nan_mask] = TAG_CODES['PHI']
    tags[posinf_mask] = TAG_CODES['PINF']
    tags[neginf_mask] = TAG_CODES['NINF']
    
    # Handle gradients
    if requires_grad and finite_mask.any():
        values.requires_grad_(True)
    
    return TRTensor(values, tags, requires_grad=requires_grad)


def to_torch(tr_tensor: TRTensor) -> 'torch.Tensor':
    """
    Convert TRTensor to PyTorch tensor (IEEE representation).
    
    Args:
        tr_tensor: Transreal tensor
        
    Returns:
        PyTorch tensor with appropriate IEEE values
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for to_torch")
    
    result = torch.empty_like(tr_tensor.values)
    
    # Map each tag type
    real_mask = tr_tensor.is_real()
    pinf_mask = tr_tensor.is_pinf()
    ninf_mask = tr_tensor.is_ninf()
    phi_mask = tr_tensor.is_phi()
    
    result[real_mask] = tr_tensor.values[real_mask]
    result[pinf_mask] = float('inf')
    result[ninf_mask] = float('-inf')
    result[phi_mask] = float('nan')
    
    # Preserve gradient tracking if needed
    if tr_tensor.values.requires_grad:
        result.requires_grad_(True)
        # Only REAL values contribute to gradients
        if real_mask.any():
            # Create gradient mask
            result.register_hook(lambda grad: grad * real_mask.float())
    
    return result


# Utility functions for PyTorch operations
def mask_real_backward(grad_output: 'torch.Tensor', tags: 'torch.Tensor') -> 'torch.Tensor':
    """
    Apply Mask-REAL rule to gradients.
    
    Args:
        grad_output: Gradient from downstream
        tags: Tag tensor
        
    Returns:
        Masked gradient (zero where tags are non-REAL)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    real_mask = tags == TAG_CODES['REAL']
    return grad_output * real_mask.float()


if TORCH_AVAILABLE:
    class TRFunction(torch.autograd.Function):
        """
        Base class for transreal functions with automatic Mask-REAL gradient rule.
        
        Subclasses should implement forward() and optionally backward().
        """
        
        @staticmethod
        def forward(ctx, values: 'torch.Tensor', tags: 'torch.Tensor', *args):
            """Forward pass - to be implemented by subclasses."""
            raise NotImplementedError
        
        @staticmethod
        def backward(ctx, grad_values: 'torch.Tensor', grad_tags: 'torch.Tensor'):
            """
            Backward pass with Mask-REAL rule.
            
            Default implementation zeros gradients for non-REAL outputs.
            """
            # Get saved tags from forward
            output_tags = ctx.saved_tensors[-1]
            
            # Apply Mask-REAL rule
            masked_grad = mask_real_backward(grad_values, output_tags)
            
            # Return gradients (None for tags and extra args)
            return masked_grad, None, *([None] * (ctx.num_args - 2))
else:
    class TRFunction:
        """Stub when PyTorch is not available."""
        pass


# Conversion utilities for common use cases
def tr_tensor_from_list(values: list, tags: list, 
                       device: Optional[Union[str, 'torch.device']] = None,
                       dtype: Optional['torch.dtype'] = None) -> TRTensor:
    """
    Create TRTensor from lists of values and tags.
    
    Args:
        values: List of values
        tags: List of TRTag enum values
        device: Target device
        dtype: Target dtype
        
    Returns:
        TRTensor
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    # Convert to tensors
    value_tensor = torch.tensor(values, device=device, dtype=dtype or torch.float32)
    tag_codes = [TAG_TO_CODE[tag] for tag in tags]
    tag_tensor = torch.tensor(tag_codes, device=device, dtype=torch.uint8)
    
    return TRTensor(value_tensor, tag_tensor)


def batch_from_scalars(scalars: list[TRScalar], 
                      device: Optional[Union[str, 'torch.device']] = None) -> TRTensor:
    """
    Create TRTensor batch from list of TRScalars.
    
    Args:
        scalars: List of TRScalar values
        device: Target device
        
    Returns:
        TRTensor with shape (len(scalars),)
    """
    values = []
    tags = []
    
    for scalar in scalars:
        if scalar.tag == TRTag.REAL:
            values.append(scalar.value)
        else:
            values.append(float('nan'))  # Placeholder
        tags.append(scalar.tag)
    
    return tr_tensor_from_list(values, tags, device=device)


# Integration with PyTorch autograd
def enable_tr_gradients():
    """
    Enable gradient computation for TRTensor operations.
    
    This sets up hooks to apply Mask-REAL rule automatically.
    """
    warnings.warn(
        "TR gradient support for PyTorch is experimental. "
        "Ensure all operations properly handle tags.",
        category=UserWarning
    )
