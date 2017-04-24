import gohai.simpleimagelabeling.*;
SimpleImageLabeling sil;
PImage grace;

void setup() {
  size(517, 606);
  grace = loadImage("grace_hopper.jpg");

  sil = new SimpleImageLabeling(this);

  // analyze the image
  sil.analyze(grace);

  // iterate through the labels with high probability
  for (int i=0; i < sil.length; i++) {
    println(sil.label(i) + ": " + sil.probability(i));
  }
}

void draw() {
  image(grace, 0, 0);
}
