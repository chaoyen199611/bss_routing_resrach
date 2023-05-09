#!/bin/bash

STRDATE=1;
ENDDATE=5;
STRT="06:00:00";
ENDT="09:00:00";
AREA=("鹽埕區","前金區");

FUNCTION=$1;


python bss_routing.py $STRDATE $ENDDATE $STRT $ENDT $AREA $FUNCTION