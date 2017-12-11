import sys
import tensorflow as tf
from PIL import Image,ImageFilter

def predict(img):
    # Create the model with same dimensions, but empty values
    data = tf.placeholder(tf.float32, [None, 784])
    weight = tf.Variable(tf.zeros([500, 10]))
    bias = tf.Variable(tf.zeros([10]))
    hl_weight = tf.Variable(tf.zeros([784, 500]))
    hl_bias = tf.Variable(tf.zeros([500]))
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(data, hl_weight), hl_bias))
    hypothesis = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, weight),bias))

    saver = tf.train.Saver()

    # Recovers the model and runs img through the model to predict the value
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "model.ckpt")
        print ('Model successfully restored')
        prediction=tf.argmax(hypothesis,1)
        return prediction.eval(feed_dict={data: [img]}, session=sess)


def prepare_img(argv):

    # Prepares the pixel values of the image
    pic = Image.open(argv).convert('L')
    width = float(pic.size[0])
    height = float(pic.size[1])
    image = Image.new('L', (28,28), (255))

    # Resize either width or height to 20 pixels depending on which is bigger
    if width > height:
        height_resize = int(round((20.0/width*height), 0))
        # Minimum pixel is 1
        if height_resize == 0:
            height_resize = 1
        # resize and sharpen
        img = pic.resize((20,height_resize), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        # Horizontal position
        wtop = int(round(((28 - height_resize)/2),0)) 
        # Resized image inserted to blank canvas
        image.paste(img, (4, wtop)) 
    else:
        width_resize = int(round((20.0/height*width),0)) 
        # Minimum pixel is 1
        if width_resize == 0: 
            width_resize = 1
        # resize and sharpen
        img = pic.resize((width_resize,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        # Vertical position
        wleft = int(round(((28 - width_resize)/2),0)) 
        # Insert resized image on blank canvas
        image.paste(img, (wleft, 4)) 

    # Get the pixel valuesk
    px = list(image.getdata())
    
    # Normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in px] 
    return tva

def main(argv):
    img_val = prepare_img(argv)
    predictint = predict(img_val)
    print ("Your number is: ", predictint[0]) 
    
if __name__ == "__main__":
    main(sys.argv[1])
