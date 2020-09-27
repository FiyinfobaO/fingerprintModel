from tflite_runtime.interpreter import Interpreter
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(description='Fingerprint Classification')
parser.add_argument('--filename', type=str, help='Specify the filename', required=True)
parser.add_argument('--model_path', type=str, help='Specify the model path', required=True)
parser.add_argument('--label_path', type=str, help='Specify the label map', required=True)
parser.add_argument('--top_k', type=int, help='How many top results', default=3)

args = parser.parse_args()

filename = args.filename
model_path = args.model_path
label_path = args.label_path
top_k_results = args.top_k

with open(label_path, 'r') as f:
    labels = list(map(str.strip, f.readlines()))

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read image
img = cv2.imread(filename)
print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img.shape)

# Get input size
input_shape = input_details[0]['shape']
print(input_shape)
size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

# Preprocess image
img.resize(size)
print(img.shape)
img = np.array(img, dtype=np.float32)
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
print(img.shape)
# normalize the image
img = img / 255.0

# Add a batch dimension
input_data = np.expand_dims(img, axis=0)
print(input_data.shape)

# Point the data to be used for testing and run the interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtain results and map them to the classes
predictions = interpreter.get_tensor(output_details[0]['index'])[0]
#predicted_label = np.argmax(predictions)
#print(labels[predicted_label])

# # Get indices of the top k results
top_k_indices = np.argsort(predictions)[::-1][:top_k_results]
#
for i in range(top_k_results):
    print(labels[top_k_indices[i]], predictions[top_k_indices[i]] / 255.0)

