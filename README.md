# CS 434 Machine Learning

This repo is to workthrough the book _Genetic Algorithms and Machine Learning for Programmers_ by Frances Buontempo.

## Escape!

In this chapter, we set up a `hello_world` to make sure we can draw a _paper bag_ for us to code our way out of, and an `escape` program in python to _escape_ the paper bag and log some data to give us a dataset to work with in later chapters. Below are some examples of how to run `escape.py`.

```
python escape.py --function=line
python escape.py --function=triangles --number=8
python escape.py --function=squares --number=40
python escape.py --function=spirangles
```

## Decide!

In this chapter, we used supervised machine learning to locate the edges of the paper bag. We used category data, treating each coordinate as a specific value, rather than a number from a possible range. Because we knew the shape of the bag, we were able to reformulate the decision tree using numeric ranges.

```
python decide.py
```
