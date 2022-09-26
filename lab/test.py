import tensorflow as tf
from tensorflow import keras

dense = keras.layers.Dense(units=1, activation=None)
softmax = keras.layers.Softmax()

inputs = tf.random.normal([23, 30, 200])
print(inputs)

softmax_tanh = softmax(tf.squeeze(dense(inputs), axis=[-1]))
print(softmax_tanh)
print(tf.reverse(inputs, axis=[1]))
lstm = tf.keras.layers.LSTM(4)
output = lstm(inputs)
print(output.shape)
print(output)

lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
print(whole_seq_output)
print(whole_seq_output.shape)

print(final_memory_state.shape)
print(final_memory_state)

print(final_carry_state.shape)

lstm = tf.keras.layers.LSTM(4, return_state=True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
print(whole_seq_output.shape)
print(whole_seq_output.shape)

print(final_memory_state.shape)
print(final_memory_state)

print(final_carry_state.shape)


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def buildTree(self, preorder: list[int], inorder: list[int]) -> TreeNode:
        val2index = {val: index for index, val in enumerate(inorder)}

        def rebuild(preorder, inorder, root):
            if len(preorder) < 1 or len(inorder) < 1:
                return None

            root_val = preorder[0]
            index_root_in = val2index[root_val]

            root = TreeNode(root_val)

            left_in = inorder[:index_root_in]
            left_pre = preorder[1:len(left_in)]
            rebuild(left_pre, left_in, root=root.left)

            right_in = inorder[index_root_in + 1:]
            right_pre = preorder[len(left_in):]
            rebuild(right_pre, right_in, root=root.right)

        root = TreeNode(None)
        rebuild(preorder, inorder, root)

        return root

a = Solution()

b = a.buildTree([3,9,20,15,7], [9,3,15,20,7])
print(b)