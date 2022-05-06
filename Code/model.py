from keras.preprocessing import image
import numpy as np
from keras.models import load_model

model = load_model('./Data/saved_model/CNN_Cat_Dog_Model.h5')

model.summary()
# Examine Weights
model.weights

# Examine Optimizer
model.optimizer

test_image = image.load_img('./Data/single_prediction/dog.webp', target_size=(64, 64))
# Add a 3rd Color dimension to match Model expectation
test_image = image.img_to_array(test_image)
# Add one more dimension to beginning of image array so 'Predict' function can receive it (corresponds to Batch, even if only one batch)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
# We now need to pull up the mapping between 0/1 and cat/dog

# Map is 2D so check the first row, first column value
print(result)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
# Print result


print("\nPrediction: " + prediction)
