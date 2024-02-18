import numpy as np
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return (input > 0) * grad_output


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 1 / (1 + np.exp(-input))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sigmoid_output = self.compute_output(input)
        return sigmoid_output * (1 - sigmoid_output) * grad_output


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        exp_input = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exp_input / np.sum(exp_input, axis=-1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        exp_input = np.exp(input - np.max(input, axis=-1, keepdims=True))
        softmax_output = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        return softmax_output * (grad_output - np.sum(softmax_output * grad_output, axis=-1, keepdims=True))



class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        max_vals = np.max(input, axis=-1, keepdims=True)
        exp_input = np.exp(input - max_vals)
        logsumexp = np.log(np.sum(exp_input, axis=-1, keepdims=True))
        return input - max_vals - logsumexp

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        exp_input = np.exp(input)
        logsumexp = np.log(np.sum(exp_input, axis=-1, keepdims=True))
        softmax_output = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        return grad_output - softmax_output * np.sum(grad_output, axis=-1, keepdims=True)


#%%
