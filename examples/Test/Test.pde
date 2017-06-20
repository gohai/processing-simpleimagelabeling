/**
 *  This example uses a Neural Network to analyze an image
 *  and prints the three most likely labels (out of 1000)
 *  to the console when it is done.
 */

import gohai.simpleimagelabeling.*;
SimpleImageLabeling sil;
PImage cover;

void setup() {
  size(517, 606);
  cover = loadImage("cover.jpg");

  sil = new SimpleImageLabeling(this);

  // analyze the image
  sil.analyze(cover);

  // iterate through the labels with high probability
  for (int i=0; i < sil.length; i++) {
    println(sil.label(i) + ": " + sil.probability(i));
  }
}

void draw() {
  image(cover, 0, 0);
}
