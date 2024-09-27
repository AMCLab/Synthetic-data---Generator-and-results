#!/bin/bash

path="/lacey_carbon_results.sh"

(cd "reproduce_results_lacey_carbon" && ."$path")

path="/amorphous_carbon_results.sh"

(cd "reproduce_results_amorphous_carbon" && ."$path")



