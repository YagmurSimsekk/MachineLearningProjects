# AI application in health care/medical devices

ICU patients are permanently connected to medical devices that track their vital signs, such as blood pressure, heart rate, oxygen saturation and other parameters. 

These devices monitor whether the vital signs are in an acceptable, i.e., harmless range. If a parameter is too high or too low, they will issue an alarm. If there is an alarm, nurses and physicians are notified and can check the condition of the patient.

**Not every alarm is relevant.** The manufacturers of the ICU monitors are interested to hedge against being sued. Therefore the machines produce alarms very easily, e.g. even if a sensor is just temporarily moved. For every ICU patient, about 100 alarms are produced per day. About 29 of them are relevant and about three relevant alarms are missing. In general, a false alarm needs about 2 minutes and a relevant alarm 5 minutes of a physician’s time to be taken care of. **Checking on false alarms creates a lot of additional work for the ICU teams and increases the threat of
missing relevant alarms.**

# Data
Data set that was gathered in a study on intensive care data management with a German university. The data set contains 4,008 joint measurements of
basic vital signs as well as information about alarms. The features are
● hr : heart rate
● rr : respiratory rate
● spo2 : oxygen saturation measured via pulsoxymetrie
● sys : systolic blood pressure (non-invasive)
● dia : diastolic blood pressure (non-invasive)
● monitor_alarm : whether patient monitor throws an alarm
● doctor_alarm : whether the doctor would have wanted an alarm (manual evaluation)
