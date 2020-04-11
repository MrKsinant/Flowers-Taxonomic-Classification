#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#############################################################
# PROGRAMMER: Udacity, Inc.                                 #
# DATE CREATED: 18/11/2018                                  #
# REVISED DATE: -                                           #
# PURPOSE: This file contains specific code to run programs #
#          properly inside Udacity working environments     #
#############################################################


##################
# Needed imports #
##################

import signal, requests
from contextlib import contextmanager


#################
# Specific code #
#################

DELAY = INTERVAL = 4*60
MIN_DELAY = MIN_INTERVAL = 2*60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


######################
# Constant variables #
######################

def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler

@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace import active_session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)

def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable