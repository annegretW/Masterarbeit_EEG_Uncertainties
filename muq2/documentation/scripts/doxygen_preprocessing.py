import sys
import subprocess
import os
import jinja2
import glob
import pathlib
from distutils.dir_util import copy_tree


"""
This script is run during the "make doc" build just before calling doxygen.  It
can be used for generating doxygen input files.
"""

# Read the root and build directoriess
assert (len(sys.argv)>2)

baseDir = sys.argv[1]
if(baseDir[-1]=='/'):
    baseDir = baseDir[0:-1]

buildDir = sys.argv[2]
if(buildDir[-1]=='/'):
    buildDir = buildDir[0:-1]

# Create the output directory if it doesn't exist
if not os.path.exists(buildDir+'/doxygen_prep'):
    os.makedirs(buildDir+'/doxygen_prep')


def MakeInstallation():
    """
    Combines markdown files in "doxFiles/Installation" with the bash script in
    "SupportScripts" to create a single set of installation instructions.
    """
    md1 = baseDir + '/documentation/doxFiles/excluded/source_install1.md'
    md2 = baseDir + '/documentation/doxFiles/excluded/source_install2.md'
    bash = baseDir + '/SupportScripts/install-instructions.sh'

    newText = subprocess.run(["cat", md1, bash, md2], stdout=subprocess.PIPE)
    print(newText.stdout)
    with open(buildDir+'/doxygen_prep/source_install.md', 'wb') as f:
        f.write(newText.stdout)

def ExtractCodeBlocks(srcLines):
    """
        Takes a list of lines in a source file and returns a list of dict objects
        describing each @codeblock.  Each dictionary contains the following keys:
        - "language"  A string containing the highlight.js language id.
        - "name" A string with the name of the block.
        - "id" A unique id (at least to this div) that can be used to identity blocks in the html.
        - "code" A string with the code snippet.
        - "startLine" The index where the codeblock begins.
        - "endLine" The index of the line where the codeblock ends.
    """

    blocks = []
    blockInd = 0
    groupInd = 0

    lineInd = 0
    while(lineInd<len(srcLines)):

        # Check to see if this is the start of a code block
        if('@codeblock' in srcLines[lineInd]):
            blockInd += 1
            startInd = lineInd

            line = srcLines[lineInd].split('@codeblock')[1]
            attrInds = line.find("{"), line.find(","), line.find("}")
            language = line[ attrInds[0]+1 : attrInds[1]].strip()
            name = line[ attrInds[1]+1 : attrInds[2]].strip()

            code = line[attrInds[2]+1:]
            lineInd += 1

            # Read the rest of the code block
            while('@endcodeblock' not in srcLines[lineInd][:]):
                code += '\n'
                code += srcLines[lineInd]
                lineInd += 1

                if(lineInd==len(srcLines)):
                    raise RuntimeError("Finished reading file before end of code block was found.")

            # Read the last line before the @endcodeblock
            code += srcLines[lineInd].split('@endcodeblock')[0]
            endInd = lineInd

            code = code.strip()

            blocks.append({'language':language,'id':'codeblock{}'.format(blockInd), 'name':name, 'code':code, 'startLine':startInd, 'endLine':endInd})
        lineInd += 1

    # Group the blocks so that blocks on consecutive lines are put in the same list
    blocks[0]['groupid'] = 'codegroup{}'.format(groupInd)
    groups = [[blocks[0]]]
    currInd = 0
    for block in blocks[1:]:
        if(block['startLine']==groups[currInd][-1]['endLine']+1):
            block['groupid'] = groups[currInd][0]['groupid']
            groups[currInd].append(block)
        else:
            groupInd += 1
            block['groupid'] = 'codegroup{}'.format(groupInd)
            groups.append([block])
            currInd += 1


    return groups


def ApplyTemplate(templateText, srcfile):
    """ Takes a source file (e.g., *.h, *.cpp) that will be processed by doxygen and
        replaces segments of the code between "@codeblock{id,name}" and "@endcodeblock" with
        formated HTML that will be included in the doxygen output.  The mapping from
        @codeblock to html is given by a jinja2 template defined by the text in
        templateText string passed as an argument.
    """

    # convert the text into a jinja2 template
    t = jinja2.Template(templateText)
    print(srcfile, ' Template:', t)
    with open(srcfile, 'r') as fin:
        srcText = fin.read()
        srcLines = srcText.split('\n')

        # Check to see if there are any code blocks
        if('@codeblock' in srcText):
            blockGroups = ExtractCodeBlocks(srcLines)
            print(blockGroups)

            for group in blockGroups[::-1]:

                # Create the html
                newText = t.render(allblocks=group, groupid=group[0]['groupid'])

                # Put the new text on the line where the group started
                srcLines[group[0]['startLine']] = newText

                # Clear the rest of the lines.  We're working backward through
                # the file, so this should mess indices of earlier groups
                del srcLines[group[0]['startLine']+1 : group[-1]['endLine']+1]

            newFileContent = '\n'.join(srcLines)

            return newFileContent
        else:
            return srcText

def ProcessFiles():
    """
    """

    fileTypes = ['.h','.cpp','.md','.dox']
    excludeDirs = ['external', 'build', 'cmake', 'joss', 'documentation/doxFiles/excluded/']

    # Read the template file
    with open('../documentation/scripts/codeblock_template.html', 'r') as fin:
        template = fin.read()

    # Figure out which files in the source directory we will prepare for doxygen to parse
    files = []
    for type in fileTypes:
        possibleFiles = glob.glob(baseDir + '/**/*' + type, recursive = True)

        for file in possibleFiles:
            include = True
            for dir in excludeDirs:
                if(file.startswith(baseDir + '/' + dir)):
                    include = False
                    break

            if(include):
                files.append(file)


    # Process each file and copy to the build directory
    for file in files:
        newContent = ApplyTemplate(template,file)

        newFilename = buildDir + '/doxygen_prep' + file[len(baseDir):]

        # Create a path to the new file it doesn't already exist
        pathlib.Path('/'.join(newFilename.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

        # Save the new file
        with open(newFilename,'w') as fout:
            fout.write(newContent)



ProcessFiles()
MakeInstallation()
copy_tree(baseDir + '/documentation/pics', buildDir + '/documentation/pics')
