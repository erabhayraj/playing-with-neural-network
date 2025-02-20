import numpy as np

def Convolution_1D(
    array1: np.array, array2: np.array, padding: str, stride: int = 1
) -> np.array:
    """
    Compute the convolution array1 * array2.
    
    Args:
    array1: np.array, input array
    array2: np.array, kernel
    padding: str, padding type ('full' or 'valid')
    stride: int, specifies how much we move the kernel at each step
    
    Carefully look at the formula for convolution in the problem statement, specifically the g[n-m] term.
    What does it indicate?
    Also note the constraints on sizes:
    - For padding='full', the sizes of array1 and array2 can be anything
    - For padding='valid', the size of array1 must be greater than or equal to the size of array2.
    
    Returns:
    np.array, output of the convolution
    """
    output=[]
    m=array1.shape[0]
    n=array2.shape[0]
    if padding == "valid":
        output_len=m-n+1
        output=np.zeros(output_len)
        for i in range(output_len):
            output[i]=np.sum(array1[i:i+n]*array2[::-1])
    elif padding=="full":
        cnt=n-1
        new_m=2*cnt+m
        updated_array1=np.zeros(new_m)
        updated_array1[cnt:cnt+m]=array1
        output_len=new_m-n+1
        output=np.zeros(output_len)
        for i in range(output_len):
            output[i]=np.sum(updated_array1[i:i+n]*array2[::-1])

    return output

def probability_sum_of_faces(p_A: np.array, p_B:np.array) -> np.array:
    """
    Compute the probability of the sum of faces of two unfair dice rolled together.

    Args:
        p_A (np.array): Probabilities of the faces of die A, from 1 to number of faces of die A
        p_B (np.array): Probabilities of the faces of die B, from 1 to number of faces of die B

    Returns:
        np.array: Probabilities of the sum of faces of die A and die B.
    
    Note that the sum of the faces cannot be 1, and hence the probability array should start from 2.
    For example, given two fair dice with 2 faces each,
    p_A = [0.5, 0.5] and p_B = [0.5, 0.5], the probabilities of sum of faces are

    Sum = 1 => Probability = 0
    Sum = 2 => Probability = 0.25 (1_A, 1_B)
    Sum = 3 => Probability = 0.5 (1_A, 2_B), (2_A, 1_B)
    Sum = 4 => Probability = 0.25 (2_A, 2_B)

    The output should start with the probability of 2, and hence the expected output is [0.25, 0.5, 0.25].
    """
    m = p_A.shape[0]
    n = p_B.shape[0]
    output_probs=np.zeros(m+n)
    for i in range(1,m+1):
        for j in range(1,n+1):
            output_probs[i+j-1]+=p_A[i-1]*p_B[j-1]
    return output_probs
    

if __name__ == "__main__":
    array1=np.array([0.5, 0.5])
    array2=np.array([0.5, 0.5])
    output=Convolution_1D(array1=array1,array2=array2,padding="valid",stride=1)
    print(output)
    output=probability_sum_of_faces(array1,array2)
    print(output)