/* -*- mode: java; c-basic-offset: 2; indent-tabs-mode: nil -*- */

/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 * Ported to Processing by Gottfried Haider 2017.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package gohai.simpleimagelabeling;

import java.nio.ByteBuffer;
import java.util.Arrays;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import processing.core.*;


/**
 *  XXX
 */
public class SimpleImageLabeling {

  private static final String graphDefFn = "tensorflow_inception_graph.pb";
  private static final String labelsFn = "imagenet_comp_graph_label_strings.txt";
  private static final int maxLabels = 10;

  protected PApplet parent;
  // loaded from file
  protected byte[] graphDef;
  protected String[] allLabels;
  // calculated
  public int length;
  protected float[] probabilities;
  protected String[] labels;


  /**
   *  Create a new image labeling instance
   */
  public SimpleImageLabeling(PApplet parent) {
    this.parent = parent;

    System.out.println("Using TensorFlow " + TensorFlow.version());
    // attempt to load the default model
    loadModel(graphDefFn, labelsFn);
  }

  /**
   *  XXX
   */
  public void loadModel(String pb, String labels) {
    try {
      graphDef = parent.loadBytes(pb);
      allLabels = parent.loadStrings(labels);
    } catch (Exception e) {
    }
  }

  /**
   *  XXX
   */
  public void analyze(PImage img) {
    if (graphDef == null || allLabels == null) {
      System.err.println("You need to download");
      System.err.println("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip");
      System.err.println("and place the extracted files in the sketch's data folder");
      throw new RuntimeException("No model files found");
    }

    // XXX: faster way to convert to uint8 RGB byte-array?
    // ~15ms on Macbook Air
    byte[] imageBytes = new byte[img.height*img.width*3];
    img.loadPixels();
    for (int i=0; i < img.pixels.length; i++) {
        imageBytes[i*3+0] = (byte)((img.pixels[i] >> 16) & 0xff);  // R
        imageBytes[i*3+1] = (byte)((img.pixels[i] >> 8) & 0xff);   // G
        imageBytes[i*3+2] = (byte)(img.pixels[i] & 0xff);          // B
    }

    // construct and execute graph
    // ~ 553ms on Macbook Air
    Tensor image = constructAndExecuteGraphToNormalizeImage(imageBytes, img.width, img.height);
    float[] labelProbabilities = executeInceptionGraph(graphDef, image);

    // create sorted arrays of labels and their respective probabilities
    length = (labelProbabilities.length < maxLabels) ? labelProbabilities.length : maxLabels;
    labels = new String[length];
    probabilities = new float[length];
    for (int i=0; i < labelProbabilities.length; i++) {
      for (int j=0; j < length; j++) {
        if (probabilities[j] < labelProbabilities[i]) {
          // insert and shift downwards
          if (j < length-1) {
            System.arraycopy(labels, j, labels, j+1, length-j-1);
            System.arraycopy(probabilities, j, probabilities, j+1, length-j-1);
          }
          labels[j] = allLabels[i];
          probabilities[j] = labelProbabilities[i];
          break;
        }
      }
    }
  }

  /**
   *  XXX
   */
  public int length() {
    return length;
  }

  /**
   *  XXX
   */
  public String label(int index) {
    if (length <= index) {
      throw new RuntimeException("You need to call analyze(PImage) before you can retrieve labels");
    }
    return labels[index];
  }

  /**
   *  XXX
   */
  public float probability(int index) {
    if (length <= index) {
      throw new RuntimeException("You need to call analyze(PImage) before you can retrieve probabilities");
    }
    return probabilities[index];
  }


  /*
   *  Largely unmodified TensorFlow example code below
   */

  private static Tensor constructAndExecuteGraphToNormalizeImage(byte[] imageBytes, int width, int height) {
    try (Graph g = new Graph()) {
      GraphBuilder b = new GraphBuilder(g);
      // Some constants specific to the pre-trained model at:
      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
      //
      // - The model was trained with images scaled to 224x224 pixels.
      // - The colors, represented as R, G, B in 1-byte each were converted to
      //   float using (value - Mean)/Scale.
      final int H = 224;
      final int W = 224;
      final float mean = 117f;
      final float scale = 1f;

      // convert the imageBytes (1D array of uint8) to a 3D tensor
      final long[] shape = {height, width, 3};
      final Tensor image = Tensor.create(DataType.UINT8, shape, ByteBuffer.wrap(imageBytes));

      // Since the graph is being constructed once per execution here, we can use a constant for the
      // input image. If the graph were to be re-used for multiple input images, a placeholder would
      // have been more appropriate.
      final Output input = b.constantTensor("input", image);
      final Output output =
          b.div(
              b.sub(
                  b.resizeBilinear(
                      b.expandDims(
                          // no need to decode the JPEG, start by converting the uint8 to floats
                          b.cast(input, DataType.FLOAT),
                          b.constant("make_batch", 0)),
                      b.constant("size", new int[] {H, W})),
                  b.constant("mean", mean)),
              b.constant("scale", scale));
      try (Session s = new Session(g)) {
        return s.runner().fetch(output.op().name()).run().get(0);
      }
    }
  }

  private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef);
      try (Session s = new Session(g);
          Tensor result = s.runner().feed("input", image).fetch("output").run().get(0)) {
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != 1) {
          throw new RuntimeException(
              String.format(
                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                  Arrays.toString(rshape)));
        }
        int nlabels = (int) rshape[1];
        return result.copyTo(new float[1][nlabels])[0];
      }
    }
  }


  // In the fullness of time, equivalents of the methods of this class should be auto-generated from
  // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
  // like Python, C++ and Go.
  static class GraphBuilder {
    GraphBuilder(Graph g) {
      this.g = g;
    }

    Output div(Output x, Output y) {
      return binaryOp("Div", x, y);
    }

    Output sub(Output x, Output y) {
      return binaryOp("Sub", x, y);
    }

    Output resizeBilinear(Output images, Output size) {
      return binaryOp("ResizeBilinear", images, size);
    }

    Output expandDims(Output input, Output dim) {
      return binaryOp("ExpandDims", input, dim);
    }

    Output cast(Output value, DataType dtype) {
      return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
    }

    Output decodeJpeg(Output contents, long channels) {
      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
          .addInput(contents)
          .setAttr("channels", channels)
          .build()
          .output(0);
    }

    // this was added to accomodate passing a custom tensor argument
    Output constantTensor(String name, Tensor t) {
      return g.opBuilder("Const", name)
          .setAttr("dtype", t.dataType())
          .setAttr("value", t)
          .build()
          .output(0);
    }

    Output constant(String name, Object value) {
      try (Tensor t = Tensor.create(value)) {
        return g.opBuilder("Const", name)
            .setAttr("dtype", t.dataType())
            .setAttr("value", t)
            .build()
            .output(0);
      }
    }

    private Output binaryOp(String type, Output in1, Output in2) {
      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
    }

    private Graph g;
  }
}
