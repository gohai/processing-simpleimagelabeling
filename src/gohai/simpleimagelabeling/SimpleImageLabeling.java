/* -*- mode: java; c-basic-offset: 2; indent-tabs-mode: nil -*- */

/*
 * Copyright (c) Limor Fried/Ladyada for Adafruit Industries, with
 * contributions from the open source community. Originally based on
 * Thermal library from bildr.org. License: MIT.
 * Ported to Processing by Gottfried Haider 2017.
 */

package gohai.simpleimagelabeling;

import org.tensorflow.TensorFlow;
import processing.core.*;


/**
 *
 */
public class SimpleImageLabeling {

  protected PApplet parent;


  /**
   *  Create a new image labeling instance
   */
  public SimpleImageLabeling(PApplet parent) {
    this.parent = parent;
    System.out.println("Hello from " + TensorFlow.version());
  }
}
