---
title: Ergo Gestures
subtitle: A catalogue of all gestures known by the Ergo project
author: 
- Boyd Kane (26723077)
---
# Ergo Gestures
What follows is first a definition of the terminology used to describe the
various gestures (taken from medical literature) and then a list of gestures
that the models will be trained to recognise.

## Terminology used for cataloguing the gestures
The language used to describe gestures is taken from medical literature. For
the readers that don't have a background in medicine, a plain English
explanation is provided here.

- Movements of the forearm
    - **Pronation**: When the forearm is rotated so that the palm faces
      downwards.
    - **Supination** When the forearm is rotated so that the palm faces
      upwards.
- Movements of the wrist or fingers
    - **Extension**: When the wrist or finger is rotated to bring the back of the
      hand closer to the outer forearm.
    - **Hyperextension**: When the finger is rotated above the plane formed by
      the palm
    - **Flexion**: When the wrist or finger is rotated to bring the palm closer to
      the inner forearm.
    - **Radial deviation**: When the wrist or finger is rotated to bring the thumb
      towards the forearm.
    - **Ulnar deviation**: When the wrist or finger is rotated to bring the pinky
      finger towards the forearm.
- Movements of the fingers
    - **Abduction**: When the fingers are splayed away from each other.
    - **Adduction**: When the fingers are brought together in a flat plane.
- **Palmar**: The area of the hand closer to the palm than to the back of the
  hand.

## Candidate gestures

There are several defined dimensions along which a gesture can change. Each
dimension is listed below, along with its possible values. Note that not every
combination is physically possible, this is merely an upper bound.

- Handedness (2): right or left
- Fingers (31): any of $_5C_{\{1, \dots, 5\}} = 5 + 10 + 10 + 5 + 1 =
  31$ for each of the combinations of 1, 2, 3, 4, or 5 fingers on each hand.
- Motion (3): flexion, extension, hyperextension
- Contact (3): No contact, thumb touches finger pad, thumb touches finger nail

Which totals to 558 different gestures, excluding any gestures made between two
hands

