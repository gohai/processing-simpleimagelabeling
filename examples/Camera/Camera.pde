/**
 *  This requires the GL Video library to be installed.
 *
 *  For use with the Raspberry Pi camera, make sure the camera is
 *  enabled in the Raspberry Pi Configuration tool and add the line
 *  "bcm2835_v4l2" (without quotation marks) to the file
 *  /etc/modules. After a restart you should be able to see the
 *  camera device as /dev/video0.
 */

import gohai.glvideo.*;
import gohai.simpleimagelabeling.*;

GLCapture video;
SimpleImageLabeling sil;

void setup() {
  fullScreen(P2D);
  noCursor();
  textSize(24);

  // first camera at 320x240 with 1fps
  video = new GLCapture(this, GLCapture.list()[0]);
  video.play();

  sil = new SimpleImageLabeling(this);
}

void draw() {
  background(0);
  if (video.available()) {
    video.read();
    sil.analyze(video);
  }

  image(video, 0, 0, width, height);
  // overlay top 3 labels
  for (int i=0; i < sil.length && i < 3; i++) {
    text(sil.label(i), 10, 34+i*31);
  }
}
