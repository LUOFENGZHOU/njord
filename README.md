# njordtoolbox

The njordtoolbox package groups usefull routines developped under the njord project.

## Requirements

To use the njordtoolbox, the following packages are required,

    * numpy
    * pandas
    * torch

If you wish to install the njordtoolbox package using pip, go directly to the next section. The setup.py file will do the requirement work for you. 
If you don't, make sure the above listed packages are installed. If you are not sure, take the following steps.

If you have conda installed on your system, you can check as:

    $ conda list

Install numpy using pip from the terminal:

	$ pip install numpy

Install pandas using pip from the terminal:

	$ pip install pandas

Install torch using pip from the terminal:

    $ pip install torch

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

The record class makes it easy to keep track of timeseries. 
It behaves like a queue with the regular methods implemented 
like appendleft, append, popleft, pop, clear, etc. 
To record a value, keep in mind you must provide the timestamp together with the value. 

An example would be:

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

## History

The history class makes it easy to manipulate historical data presented in the form of a pandas DataFrame. The index column is a datetime64 object. 

An example would be:

```
>>> import random
>>> import time
>>> import datetime
>>> from njordtoolbox import History
>>> history = History

```

## Trades

The trades class makes it easy to record trades. A trade has the characterised by a timestamp, price, base quanty, quote quanty, trading fees and an order type. The trades class makes it possible to append trades with the above listed characteristics as a regular lits.

As an example. Import first the usefull packages:

```
>>> import random
>>> import datetime
>>> from njordtoolbox import Trades
```

Then creates an empty trade object.

```
>>> trades = Trades(name="myTrades")
```

Fill the object with random new trades, for instance.

```
for i in range(20):
    timestamp = datetime.datetime.now()
    price = random.normalvariate(1.0, 0.05)
    base = random.normalvariate(100.0, 1.0)
    quote = price * base
    fees = base * 0.001
    otype = ["buy", "sell"][random.randrange(0, 2)]
    trades.append(timestamp, price, base, quote, fees, otype)
```

Display the recorded trades.

```
print(trades.aspandas())
```

That's it.
