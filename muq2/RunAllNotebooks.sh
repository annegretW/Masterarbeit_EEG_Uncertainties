#!/bin/bash

exit_code=0

summary=""

# Make sure globstar is enabled
shopt -s globstar
for notebook in **/*.ipynb; do # Whitespace-safe and recursive
    echo "======================"
    echo "Running $notebook:"
    echo ""

    papermill --progress-bar --cwd "$(dirname "$notebook")" "$notebook" - > /dev/null

    papermill_exit=$?

    echo ""

    if [ $papermill_exit -eq 0 ]
    then
        echo "Notebook $notebook successful."
	summary="$summary\nNotebook $notebook successful"
    else
        echo "Notebook $notebook FAILED!"
	summary="$summary\nNotebook $notebook FAILED!"
    fi

    exit_code=$(($exit_code + $papermill_exit)) # Simply add exit codes; will be zero if all tests successful
done


echo "======================"

echo -e $summary

echo ""

if [ $exit_code -eq 0 ]
then
    echo "All notebooks successful."
else
    echo "Some notebooks FAILED!"
fi

exit $exit_code
