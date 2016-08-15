from deepgaze.bayes_filter import DiscreteBayesFilter
import numpy as np


my_filter = DiscreteBayesFilter(10)

belief = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

cpt_motion_model = np.array([[0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1],
                             [0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0.1, 0.8, 0.1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0.1, 0.8, 0.1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0.1, 0.8, 0.1],
                             [0.1, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.8]], dtype=np.float32)

cpt_measurement_accuracy = np.array([[0.6, 0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0],
                                     [0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0],
                                     [0, 0.1, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0.1, 0.8, 0.1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0.1, 0.8, 0.1, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0.1, 0.8, 0.1],
                                     [0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.8]], dtype=np.float32)

print("From unknown position moving to 1")
belief = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

for i in range(10):
    belief_predicted = my_filter.predict(belief, cpt_motion_model)
    belief_updated = my_filter.update(belief_predicted, 1, cpt_measurement_accuracy)
    belief = belief_updated
    print("Estimated state: " + str(np.argmax(belief)) + " with probability: " + str(belief[np.argmax(belief)]))

print("Moving to state 2")
for i in range(10):
    belief_predicted = my_filter.predict(belief, cpt_motion_model)
    belief_updated = my_filter.update(belief_predicted, 2, cpt_measurement_accuracy)
    belief = belief_updated
    print("Estimated state: " + str(np.argmax(belief)) + " with probability: " + str(belief[np.argmax(belief)]))

print("Moving to state 3")
for i in range(10):
    belief_predicted = my_filter.predict(belief, cpt_motion_model)
    belief_updated = my_filter.update(belief_predicted, 3, cpt_measurement_accuracy)
    belief = belief_updated
    print("Estimated state: " + str(np.argmax(belief)) + " with probability: " + str(belief[np.argmax(belief)]))

print("Some dirty measures around 3...")
for i in range(3):
    belief_predicted = my_filter.predict(belief, cpt_motion_model)
    belief_updated = my_filter.update(belief_predicted, 2, cpt_measurement_accuracy)
    belief = belief_updated
    print("Estimated state: " + str(np.argmax(belief)) + " with probability: " + str(belief[np.argmax(belief)]))
    belief_predicted = my_filter.predict(belief, cpt_motion_model)
    belief_updated = my_filter.update(belief_predicted, 3, cpt_measurement_accuracy)
    belief = belief_updated
    print("Estimated state: " + str(np.argmax(belief)) + " with probability: " + str(belief[np.argmax(belief)]))
    belief_predicted = my_filter.predict(belief, cpt_motion_model)
    belief_updated = my_filter.update(belief_predicted, 4, cpt_measurement_accuracy)
    belief = belief_updated
    print("Estimated state: " + str(np.argmax(belief)) + " with probability: " + str(belief[np.argmax(belief)]))








