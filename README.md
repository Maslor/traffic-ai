#traffic-ai

### Set Up
`pip3 install -r requirements.txt`

The data set (gtsrb folder) can be downloaded [here](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip).

### Running
`$ python traffic.py gtsrb`

### Video
[![Watch the demo video!](https://img.youtube.com/vi/n2qdpl8JX0Y/mqdefault.jpg)](https://youtu.be/n2qdpl8JX0Y)

### Findings

In all my attempts I kept the compilation parameters constant.

````
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
````

At first I attempted a model using the ReLu activation function and a hidden layer of 64 nodes, once again with the ReLu activation function. The accuracy was only around 10-15%.

Then I removed the hidden layer altogether and used a softmax activation function. The accuracy went up to around 80%.

I then added a hidden layer with 8 nodes and a softmax activation function. Immediately, the accuracy dropped to 5%. Same thing once I increased the number of nodes to `NUM_CATEGORIES`

Then I replaced the hidden layer by a convolutional layer, which improved the accuracy to 90%. I also added a max pooling layer which increased the accuracy to 93%.

Doubling the number of filters on the convolution layer didn’t significantly improve the accuracy.

I changed the convolutional layer’s activation function to sigmoid, which increased the accuracy to 98%

I added a hidden layer `8 * NUM_CATEGORIES` nodes but the accuracy actually dropped to 94%

I found that increasing other parameters to get accuracy over 97-98% dramatically decreased performance so the cost seemed higher than the improvement.

I also added dropout.