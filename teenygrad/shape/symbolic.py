# Symbolic integer type alias
#
# This type alias represents a symbolic integer that can be used in tensor shapes.
# In this minimal implementation, it's just an alias for the built-in int type.
#
# In more advanced implementations, this would support symbolic shape variables
# that can represent unknown dimensions at graph construction time (e.g., batch size).
# For example: sint could be Union[int, SymbolicVar] where SymbolicVar represents
# an algebraic expression over shape variables.
#
# Examples where symbolic shapes are useful:
#   - Dynamic batch sizes: shape = (batch_size, 128, 128, 3) where batch_size is symbolic
#   - Variable sequence lengths: shape = (batch_size, seq_len, hidden_dim)
#   - Shape polymorphism: write code that works for any dimension size
sint = int