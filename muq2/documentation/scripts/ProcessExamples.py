"""
PURPOSE:
    Prepares the examples for the MUQ website.
"""
import glob
import yaml
import os
import os.path

import shutil

from Cpp2Html import TutorialDocument
from NotebookToHtml import ProcessNotebook
from ScriptToHtml import ProcessScript

def CheckFields(data,filename):
    """
    Checks to make sure all required fields are contained in the "data" dictionary.

    Returns True if everything is good.  Prints a warning andd returns Falses otherwise.
    """
    requiredFields = ['title', 'tag', 'file', 'description', 'doc_level']

    for field in requiredFields:
        if(field not in data):
            print('WARNING: Field %s not present in an example from file %s.  Example will be skipped.'%(field,filename))
            print(data)
            return False

    return True


def CreateJekyllHeader(exMeta,language):
    """
    Creates the header information for the Jekyll page given the metadata for an example.
    """
    head = """---
title: {}
layout: default
description: {}
language: {}
tag: {}
doc_level: {}
---
<h1><small class="text-muted">Example</small> </br> {}<h1>
<blockquote class="blockquote"><p class="mb-0">{}</p></blockquote>
</br>
""".format(exMeta['title'], exMeta['description'],language, exMeta['tag'],exMeta['doc_level'], exMeta['title'], exMeta['description'])

    return head

def WriteFile(htmlString,outputFile):
    """
    Writes a string containing the example HTML to a specified folder.  If the
    folder doesn't exist, it is created.
    """
    # Create the directory if it doesn't exist
    folder = '/'.join(outputFile.split('/')[0:-1])
    os.makedirs(folder, exist_ok=True)

    # Write to the file
    with open(outputFile,'w') as f:
        f.write(htmlString)



def ProcessExamples(examplePath, outputPath):

    # Find all of the meta.yml files in the examples folder
    if(examplePath[-1]=='/'):
        examplePath = examplePath[0:-1]

    if(outputPath[-1]=='/'):
        outputPath = outputPath[0:-1]

    metaFiles = glob.glob(examplePath + '/**/meta.yml',recursive=True)

    # Loop through all the meta.yml files
    for filename in metaFiles:

        with open(filename) as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

            # Loop through all the examples in this file
            for temp in meta:
                if('example' in temp):
                    exMeta = temp['example']
                    if(CheckFields(exMeta, filename)):

                        # Check to make sure the file exists
                        inputPath = '/'.join(filename.split('/')[0:-1])
                        exFile = inputPath + '/' + exMeta['file']

                        outFile = outputPath + exFile[len(examplePath):]
                        outFile = '.'.join(outFile.split('.')[0:-1]) + '.html'
                        print('Generating: ', outFile)
                        html=None

                        filetype = exFile.split('.')[-1]
                        if(filetype=='ipynb'):
                            html = CreateJekyllHeader(exMeta,'python')
                            html += '\n\n'
                            html += ProcessNotebook(exFile)

                        elif(filetype=='py'):
                            html = CreateJekyllHeader(exMeta,'python')
                            html += '\n\n'
                            html += ProcessScript(exFile)

                        elif(filetype=='cpp'):
                            html = CreateJekyllHeader(exMeta,'c++')
                            html += '\n\n'
                            html += TutorialDocument(exFile).ToHTML()

                        # Write the html string to file
                        if(html is not None):
                            WriteFile(html, outFile)

                        # Copy any extra files to the output folder
                        if('other_files' in exMeta):
                            files = exMeta['other_files'].split(',')
                            for file in files:
                                otherOutName = '/'.join(outFile.split('/')[0:-1])+'/'
                                otherOutName += file.strip().strip(',')

                                otherInName = inputPath + '/'
                                otherInName += file.strip().strip(',')

                                if(os.path.exists(otherInName)):
                                    shutil.copyfile(otherInName, otherOutName)


if __name__=='__main__':
    ProcessExamples('../examples', 'webExamples')
