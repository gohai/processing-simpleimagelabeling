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
