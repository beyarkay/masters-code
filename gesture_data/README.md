# Gesture data

This directory contains all the raw, unprocessed gesture data. The structure
is:

```
gesture_data/
├── README.md
├── test/
│   └── gesture{0001..}/
│       └── {timestamp}.txt
├── train/
│   └── gesture{0001..}/
│       └── {timestamp}.txt
└── valid/
    └── gesture{0001..}/
        └── {timestamp}.txt
```

Each gesture is given an index between 0001 and 9999, although not all of them
are used in practice.

## Terminology

Describing hand motions is tricky without having proper words for how your
fingers rotate at each joint. Below is a quick explanation for what the medical
terminology means:

TODO include images of these motions

Parts of the hand:

- *Distal Interphalangeal (DIP) Joint*: the joint of the fingers (but not the
  thumb) closest to the tip of your finger.
- *Proximal Interphalangeal (PIP) Joint*: The joint of the fingers (but not
  the thumb) closest to the base of your finger.
- *Interphalangeal (IP) Joint*: The joint of your thumb closest to the tip of
  the thumb.
- *Metacarpophalangeal (MCP) Joint*: The joint of the fingers or thumb where
  the finger / thumb joins on to the base of your hand.
- *Carpometacarpal (CMC) Joint*: The joint where your thumb connects to the
  base of your wrist.

Directions of the hand:

- *Palmar*: Towards the palm of the hand.
- *Dorsal*: Away from the palm of the hand.
- *Radial*: In the direction of little finger to thumb.
- *Ulnar*:  In the direction of thumb to little finger.

Motions of the hand:

- *Pronation (supination)*: Rotation of the hand at the wrist in a palmar
  (dorsal) direction.
- *Radial (ulnar) deviation*: Rotation of the hand at the wrist in a radial
  (ulnar) direction.
- *Flexion (extension)*: Rotation of the fingers in the palmar (dorsal)
  direction 
- *Abduction (adduction)*: Splaying the fingers (closing up the fingers)
- *Palmar abduction*: moving the thumb in a palmar direction

### Resting position

A resting position is needed to indicate that no keystrokes are to be sent.
This resting position will be with both hands pointing forwards, bent at the
elbows, with the thumbs upwards and little fingers downwards. The fingers are
slightly bent. The fingers have a slight gap between them, to facilitate
movement.

TODO include a picture of this

## List of gestures
This section contains all the defined gestures, a description of them, and
their index.

### `gesture0255` - The resting position 

Binary range: `0b 1111 1111`

This gesture isn't really a gesture at all, but is the label corresponding to
'no intentional gesture made'. This is a little or no movement gesture, but
could also correspond to the hands moving as a consequence of the rest of the
body moving

### `gesture0001..gesture0010` - Single finger PIP flexions 

Binary range: `0b 0000 0001` to `0b 0000 1010`

These gestures are all single finger gestures. They each consist of one finger
making a quick jerking gesture inward towards your palm and then going back to
the resting position. This jerking motion should be done at the IP joint (for
the thumb) or the PIP joint (for the fingers).

1.  Left hand thumb IP flexion
2.  Left hand index finger PIP flexion
3.  Left hand middle finger PIP flexion
4.  Left hand ring finger PIP flexion
5.  Left hand little finger PIP flexion
6.  Right hand thumb IP flexion
7.  Right hand index finger PIP flexion
8.  Right hand middle finger PIP flexion
9.  Right hand ring finger PIP flexion
10. Right hand little finger PIP flexion

### `gesture0011..gesture0020` - Single finger PIP extensions

Binary range: `0b 0000 1011` to `0b 0001 0100`

These are similar to the single finger PIP flexions, except the motion is away
from the palm instead of towards it.

These gestures are all single finger gestures. They each consist of one finger
making a quick jerking gesture outwards away from your palm and then going back
to the resting position. This jerking motion should be done at the IP joint
(for the thumb) or the PIP joint (for the fingers). The order of the fingers
is:

11. Left hand thumb IP extension
12. Left hand index finger PIP extension
13. Left hand middle finger PIP extension
14. Left hand ring finger PIP extension
15. Left hand little finger PIP extension
16. Right hand thumb IP extension
17. Right hand index finger PIP extension
18. Right hand middle finger PIP extension
19. Right hand ring finger PIP extension
20. Right hand little finger PIP extension

### `gesture0021..gesture0030` - Single finger MCP flexions 

Binary range: `0b 0001 0101` to `0b 0001 1110`

These gestures are all single finger gestures. They each consist of one finger
making a quick jerking gesture inward towards your palm and then going back to
the resting position. This jerking motion should be done at the CMC joint (for
the thumb) or the MCP joint (for the fingers).

21.  Left hand thumb CMC flexion
22.  Left hand index finger MCP flexion
23.  Left hand middle finger MCP flexion
24.  Left hand ring finger MCP flexion
25.  Left hand little finger MCP flexion
26.  Right hand thumb CMC flexion
27.  Right hand index finger MCP flexion
28.  Right hand middle finger MCP flexion
29.  Right hand ring finger MCP flexion
30. Right hand little finger MCP flexion

### `gesture0031..gesture0040` - Single finger MCP extensions

Binary range: `0b 0001 1111` to `0b 0010 1000`

These are similar to the single finger MCP flexions, except the motion is away
from the palm instead of towards it.

These gestures are all single finger gestures. They each consist of one finger
making a quick jerking gesture outwards away from your palm and then going back
to the resting position. This jerking motion should be done at the CMC joint
(for the thumb) or the MCP joint (for the fingers). The order of the fingers
is:

31. Left hand thumb CMC extension
32. Left hand index finger MCP extension
33. Left hand middle finger MCP extension
34. Left hand ring finger MCP extension
35. Left hand little finger MCP extension
36. Right hand thumb CMC extension
37. Right hand index finger MCP extension
38. Right hand middle finger MCP extension
39. Right hand ring finger MCP extension
40. Right hand little finger MCP extension

### `gesture0041..gesture0050` - Thumb-finger taps

Binary range: `0b 0010 1001` to `0b 0011 0010`

These are all two finger motions. They consist of the right (or left) thumb
making a single tap against one of the fingers on your right (or left) hand.
They are numbered similarly to gestures `gesture0001..gesture0020` in that a
gesture ending in `2` uses the left index finger, a gesture ending in `8` uses
the right middle finger, etc. For this reason gestures `gesture0021` and
`gesture0026` aren't used, since they'd correspond to the thumb tapping itself.

All the tapping motions should be the finger pad of the thumb tapping the
finger pad of the corresponding finger.

41. No gesture
42. Left thumb taps left hand index finger
43. Left thumb taps left hand middle finger
44. Left thumb taps left hand ring finger
45. Left thumb taps left hand little finger
46. No gesture
47. Right thumb taps right hand index finger
48. Right thumb taps right hand middle finger
49. Right thumb taps right hand ring finger
50. Right thumb taps right hand little finger

### `gesture0051..gesture0060` - Full hand pronation, ulnar deviations, clockwise rotations

Binary range: `0b 0011 0011` to `0b 0011 1100`

These motions are all full hand rotations around the wrist.

51. Pronation at the left hand wrist
52. Ulnar deviation at the left hand wrist
53. Clockwise then counter clockwise rotation at the left hand wrist
54. No gesture
55. No gesture
56. Pronation at the right hand wrist
57. Ulnar deviation at the right hand wrist
58. Clockwise then counter clockwise rotation at the right hand wrist
59. No gesture
60. No gesture

### `gesture0061..gesture0070` - Full hand supination, radial deviations, counter clockwise rotations:

Binary range: `0b 0011 1101` to `0b 0100 0110`

These motions are all full hand rotations around the wrist.

61. Supination at the left hand wrist
62. Radial deviation at the left hand wrist
63. Counter clockwise then clockwise rotation at the left hand wrist
64. No gesture
65. No gesture
66. Supination at the right hand wrist
67. Radial deviation at the right hand wrist
68. Counter clockwise then clockwise rotation at the right hand wrist
69. No gesture
70. No gesture

### `gesture0071..gesture0080` - Thumb taps two fingers

Binary range: `??` to `???`

These motions are all the thumb tapping two fingers together.

71. No gesture
72. Left thumb to left index and left middle fingers
73. Left thumb to left middle and left ring fingers
74. Left thumb to left ring and left little fingers
75. Left thumb to left little and left index fingers
76. No gesture
77. Right thumb to right index and right middle fingers
78. Right thumb to right middle and right ring fingers
79. Right thumb to right ring and right little fingers
80. Right thumb to right little and right index fingers

81. Left thumb to left index and left ring fingers
82. Left thumb to left middle and left little fingers

86. Right thumb to right index and right ring fingers
87. Right thumb to right middle and right little fingers

91. No gesture
92. Left thumb to left index, middle, and ring fingers
93. Left thumb to left middle, ring, and little fingers
94. Left thumb to left ring, little, and index fingers
95. Left thumb to left little, index, and middle fingers
96. No gesture
97. Right thumb to right index, middle, and ring fingers
98. Right thumb to right middle, ring, and little fingers
99. Right thumb to right ring, little, and index fingers
10. Right thumb to right little, index, and middle fingers
