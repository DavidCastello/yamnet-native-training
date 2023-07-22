import tensorflow as tf
import os

def tflite_model_info(args):
    #file_path = os.path.join(args.save_path, args.tflite_file_name)
    file_path = "/home/dcastello/Soundless/tflite-yamnet-audio-classification/model_2s_015_new/esc_2s_015.tflite"
    interpreter = tf.lite.Interpreter(file_path)

    input_details = interpreter.get_input_details()
    print(input_details)

    output_details = interpreter.get_output_details()
    print(output_details)
