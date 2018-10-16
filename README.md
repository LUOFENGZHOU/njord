# njordtoolbox

The njordtoolbox package groups many usefull classes and functions for the njord project.

## Requirements

To use this package you will need the follwing packages to be installed.

    * numpy
    * pandas
    * etc.

If you have conda installed on your system, you can verify as:

    $ conda list

If needed, you can install the required packages using pip.

Install numpy using pip from the terminal:

	$ pip install numpy

Install pandas using pip from the terminal:

	$ pip install pandas

## Installation

Now we can install the package locally (for use on our system), with:

	$ pip install .

We can also install the package with a symlink, so that changes to the source files will be immediately available to other users of the package on our system:

	$ pip install -e .

Anywhere else in our system using the same Python, we can do this now:

```
>>> import njordtoolbox
```

## Record

The record class makes it easy to keep track of timeseries. It behaves like a queue with the regular methods implemented like appendleft, append, popleft, pop, clear, etc. To record a value, keep in mind you must provide the timestamp together with the value. An example would be:

```
>>> import random
>>> import time
>>> import datetime
>>> from njordtoolbox import Record
>>> myrecord = Record(name="normalvariate")
>>> for i in range(0,10):
>>>     time.sleep(1.0)
>>>     timestamp = datetime.datetime.now()
>>>     value = random.normalvariate(0.0, 1.0)
>>>     myrecord.append(timestamp, value)
>>> print(myrecord.t)
>>> print(myrecord.x)
```

## Trades

The trades record class.

## History

The history class.
