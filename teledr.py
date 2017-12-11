# -*- coding: utf-8 -*-

#================= Imports =================#        
#from __future__ import unicode_literals  # or use u"unicode strings"

import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")

import argparse, textwrap # command line arguments

import os 
#import fnmatch
import math

import difflib


import re # for regular expressions
import itertools
from random import shuffle # shuffles lists


from io import open # for py 3.x style fopen w/ unicode encoding

# lib for matricial ops
import numpy as np, numpy
from numpy import matrix
from numpy import linalg

# lib for graphs
import igraph 
from igraph import *
from igraph import Graph, Plot
from igraph.drawing.text import TextDrawer

c=Configuration()  #igraph options
c["general.verbose"] = True


import cairo # pyCairo, igraph for python's graphical module

#from sklearn import metrics
#from scipy.misc import comb


#from unicodedata import normalize


# libs for plotting
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import colors
import matplotlib.font_manager as font_manager
from pylab import pcolor, show, colorbar, xticks, yticks, title


#================= Class define =================#        

class story_struct:
    def __init__(self):
        self.filepath = ""                      #common filepath to all files in story
        self.commonName = ""
        self.commonFolder = ""
        self.commonPath = ""
        self.rangestring = ""
        self.title = None
        self.range = [-1,-1]
        self.chapterfiles = []                  #list of all full filepaths 
        self.files = []                         # all edgelist files here 
        self.chapters = []                      # list of chapter_structs                     
        self.characters = []                    #list of all characters (detected from readfile)
        self.characterOccurrences = []          # list of lists of degrees: characters by chapters
        self.plottedCharacters = []
        self.plottedIdx = []
        self.keywords = []
        self.collections = []
        self.edgelists = []
        self.weightlists = []
        self.graphs = []                        # list of iGraph's graphs from edgelists and weightlists (one for each chapter/episode, unless AIO is selected)
        self.evcentlists = []                   # list of iGraph's eigenvector centrality lists (one list for each graph in same order)
        self.epsilon = numpy.nan                #matrix multiplier (smaller means stronger temporal links)
        self.supermatrix = []                   
        self.jointEigenvector = []              # dominant eigenvector in supermatrix
        self.conditionalEigenvector = []        #eigenvector elements divided by corresponding mlc
        self.mnc = []                           #sum of eig centralities for each node across all timelayers
        self.mlc = []                           #sum of eig centralities for each layers across all nodes in it
        self.timeAvgArray = []                  
        self.timeAvgMaxCentrality = numpy.nan
        self.freemanIndex = numpy.nan           #story graph's freeman index (related to starlikeness or diffusion of a graph)
        self.freemanIndexWhenEachAbsent = []
        self.vitalities = []                    #vitalities for each node (calculated by removing them from the story)
        self.absvitalities = []                 #vitalities in absolute values


class chapter_struct:
    def __init__(self):
        self.file = None
        self.startpoint = -1
        self.endpoint = -1
        self.headerline = None
        self.match = None
        self.regroups = []
        self.edgelist = []
        self.weightlist = []
        self.graph = None
        self.evcents = []

class collection_struct:
    def __init__(self):
        self.keyword = ""
        self.chapters = []
        

        
        
        


#================= Parameters =================#        

p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        usage='''Use "%(prog)s --help" for more information\n\n''')

p.add_argument("-f", "--filename", type=str, required=False, metavar= 'FILE',
               help="Filename without the number. Must be used with the arguments  '-from' and '-to'(i.e. aliceC to read files of the form aliceC1.txt, aliceC2.txt...)\n\n")

p.add_argument("-from", "--firstchapter", type=int, default = -1,metavar= 'INT',
        help="Integer present in first file to be analyzed (e.g. 1 if aliceC1.txt)\n\n")

p.add_argument("-to", "--lastchapter", type=int, default = -1,metavar= 'INT',
        help="Integer present in last file to be analyzed (e.g. 12 if aliceC12.txt)\n\n")

p.add_argument("-sf", "--singlefile", type=str,metavar= 'FILE',
               help="File with all chapters. Must be used with argument 'sep'\n\n")

p.add_argument("-sep", "--separator", type=str, default = None,metavar= 'REGEX',
               help=textwrap.dedent('''\
Regular expression matching separator tag for each chapter in a single file. Examples:

1) -sep '\s*\S?\s*[sS]0*1(?!\d+)[eE]0*1(?!\d+)' : only chapter 1 of season 1 (allowing any arbitrary quantity of non significant zeros: 1, 01, 001...).

2) -sep '\s*\S?\s*[sS]\d+[eE]0*1(?!\d+)' : chapter 1 of every season.

3) -sep '\s*\S?\s*[sS]0*1[eE]\d+' : every episode in season 1.

4) -sep '\s*\S?\s*[sS]0*([1-9]|10)[eE]\d+' : every chapter in seasons 1 through 10.

5) -sep '\s*\S?\s*[sS]0*[14][eE]0*1?[13](?!\d+)' : episodes 1,3,11,13 of seasons 1 and 4.

6) -sep '\s*\S?\s*[sS]0*2[eE]0*([1-9]|1[0-2])(?!\d+)' : episodes 1 to 12 of season 2.

7) -sep '\s*\S?\s*[sS]0*[1-3][eE]0*[1-6](?!\d+)' : episodes 1 to 6 of season 1 to 3.

8) -sep '# cap 1(?!\d+)' : chapter 1

9) -sep '# cap [1-3](?!\d+)' : chapters 1 to 3

NOTE: single quotes around the pattern string recommended to prevent command line issues.

'''))

p.add_argument("-kw", "--keywords", type=lambda s: unicode(s, 'utf8'), nargs = '+', default=None, metavar= 'STRINGS',
               help="""Select chapters from single file based on keywords present in the header line. Case-insensitive. Example:
Input: -kw 'christmas' 'xmas'
Effect: matches chapters '(-sep) xmas' and '(-sep) #Christmas episode'.

""")


p.add_argument("-aio", "--allInOne", action="store_true", default= False,
               help="Creates a single graph in memory for all chapters in range\n\n")

p.add_argument("-tab", "--table",  action="store_true", default=False,
               help="Creates a table comparing all characters' degrees, eigenvector (EV) centrality, betweenness, and closeness\n\n")

p.add_argument("-tabj", "--tabj",  action="store_true", default=False,
               help="Creates a table comparing all characters' degrees, eigenvector (EV) centrality, betweenness, closeness, joint centrality and conditional centrality\n\n")


p.add_argument("-gp", "--graphPlot", action="store_true", default=False,
               help="""Draws a visual representation of all chapters in range as graphs.

""")

p.add_argument("-bp", "--barPlot", type=str, nargs = '+', choices=['b', 'c', 'd'], default=[],
               help="""Plots a bar graph. Options:
d - degree
b - betweenness
c - closeness

    """)

p.add_argument("-lp", "--linePlot", type=str, nargs = '+', choices=['d','c','b','ec','jc','cc','v'], default=[],
               help="""Plots a line graph. Options:
d - degree
b - betweenness
c - closeness
ec - eigenvector centrality
jc - joint centrality
cc - conditional (normalized) centrality
v - vitalities for all characters (by comparison with the story's Freeman Index)

""")

p.add_argument("-hp", "-hm", "--heatMap", type=str, nargs = '+', choices=['d', 'b', 'c', 'ec', 'jc', 'cc'], default = [],
               help="""Plots a heat map graph. Options:
d - degree
b - betweenness
c - closeness
ec - eigenvector centrality
jc - joint centrality
cc - conditional (normalized) centrality

    """)

p.add_argument("-chars", "--featuredCharactersList", type=lambda s: unicode(s, 'utf8'), nargs = '+',default=None,metavar= 'STRINGS',
               help="""List of featured characters who'll appear in plots and receive more emphasis in data. Doesn't apply to Graph Plot (nor Bar plot yet). Case-insensitive.

Example: -chars ALICE Bob claire

NOTE: Should both -chars and -ff arguments be present, the resulting list of characters will be a disjunction of both.

""")


p.add_argument("-ff", "--featuredCharactersFile", type=str,metavar= 'FILE',
               help="""File containing a list of featured characters who'll appear in plots and receive more emphasis in data. Case-insensitive.

Example: -ff alice/plotchars.txt, containing the stream: ALICE Bob claire

NOTE: Should both -chars and -ff arguments be present, the resulting list of characters will be a disjunction of both.

""")



p.add_argument("-top", "--top", type=str, default=None, choices=['d', 'b', 'c', 'mnc'],
               help="""Automatically reduces the number of characters shown in plots by ranking them based on a chosen criterion, if no list of featured characters is inserted manually. Should be used with '--percentageCutoff'. Options:
d - degree
b - betweenness
c - closeness
mnc - marginal node centrality (sum of a character's JC across chapter range)

""")

p.add_argument("-p", "--percentageCutoff", type=float, metavar = "[0.0, 1.0]",default=0.45,
               help="""Used alongside -top and its chosen criterion, this argument indirectly determines the amount of characters plotted. If argument '--top' is called and this argument isnt, it will be defined automatically as 0.45 as default.

The input, a float between 0.0 and 1.0, is the minimum percentage that the accumulated centralities of all 'top' characters must amount to, in comparison with the accumulate centralities of all characters in the story.

Example: for -top d and -p 0.45, the criterion is degree centrality and the cutoff is at 45%%. All characters will be ranked by degree centrality in descending order and their centralities added, one by one, until the cumulative reaches 45%% of the total degree centrality.


""") #Default: %(default)s")


p.add_argument("-t", "--title", type=str, default=None,
               help="Title of the story (optional - if absent, the basename for the edgelists' file(s) will be used)\n\n")


p.add_argument("-j", "--joint", action="store_true", default=False,
               help="Calculates joint centrality and other supermatrix centralities derived from it (Conditional, Marginals, Time Averaged, Freeman Index)\n\n")

p.add_argument("-v", "--vitalities", action="store_true", default=False,
               help="Calculates the vitalities for all characters in chapter range\n\n")


p.add_argument("-ms", "--firstOrderMoverScore", action="store_true", default=False,
               help="Calculates the first-order Mover Scores for all characters in chapter range. Results on screen.\n\n")


p.add_argument("-eps", "--epsilon", type=float, required=False, default=0.00001, metavar= 'FLOAT',
        help="Scalar multiplying adjacency matrices - related to strength of temporal link between chapters for Taylor's Joint Centrality.\n\n ") #Default: eps = %(default)s")




#p.add_argument("-verify", "--verify",  action="store_true", default=False,
 #              help="Sweeps files in range for invalid edges, loops and possible typos.")


#================= Functions =================#        

def validatesChapterRange(first, last):
    
    if first < 0 or last < 0 or last<first:
        return 0;
    else:
        return 1;

def getsFilepaths(path, first, last):
    
    chapterfiles = []   # receives file names from which the graphs will be read
    missing = []        # receives any file not found in case of failure

    error = False

    print "Searching for: "
    print "------------------------------------------------"
    for i in range(first, last+1):
        file = str(path) + str(i) + '.txt'
        print file
        if os.path.isfile(file) == False:
            error = True
            missing.append(i)
        else:
            chapterfiles.append(file)
    print "------------------------------------------------"
    
    if error == True:
        print "Error. Missing chapters: ", missing
        sys.exit()

    else:
        print "All files found.\n"
        return chapterfiles;



def getsFileInfoFromUser():

    chapterfiles = []   # receives file names from which the graphs will be read
    missing = []        # receives any file not found in case of failure

    errorflag = False

    done = False

    while not done:
        folder = str(raw_input("Folder path: "))
        if not os.path.isdir(folder):
            print "Folder not found!"

        else:
            while not done:
                title = str(raw_input("Title common to all files: "))
                try:
                    first = int(raw_input("First chapter: "))
                    if first < 0:
                        raise ValueError
                    last = int(raw_input("Last chapter: "))
                    if last < 0 or last < first:
                        raise ValueError

                except ValueError:
                    print "Please provide nonnegative integers such that the last is not smaller than the first"
                else:
                    for i in range(first, last+1):
                        aux_title = title + str(i) + '.txt'
                        filepath = os.path.join(folder, aux_title)
                        print filepath
                        if os.path.isfile(filepath) == False:
                                errorflag = True
                                missing.append(i)
                        else:
                                chapterfiles.append(aux_title)
                    if errorflag == True:
                        print "Error. Missing chapters: ", missing
                        del missing[:]
                        del chapterfiles[:]
                        errorflag = False
                    else:
                        done = True

    return folder, title, chapterfiles, first, last;


def argsortAscending(array):
                
        indexes = array.argsort()

        return indexes;

def argsortDescending(array):

    indexes = array.argsort()[::-1]

    return indexes;


def makesGraphFromEdgesWeights(nnodes, edgeList, weightList):

    g = Graph()
    g.add_vertices(nnodes)
    g.add_edges( edgeList )
    g.es["weight"] = weightList

    return g;

def readsGraphFromFile(filePath, nodeList):

    edges, weights = readsEdgelistFromFile(filePath, nodeList)

    graph = makesGraphFromEdgesWeights( len(nodeList), edges, weights)

    return graph;

def getsAdjacencyFromGraph(graph):

    adjMatrix = graph.get_adjacency() # matrix received from igraph as a list of lists; requires subsequent rearrangement for correct use

    adjMatrix = numpy.array(adjMatrix.data) # lists properly arranged into matrix form

    return adjMatrix;

def getsAdjacencyMatrix(filePath, nodeList):
        
    # read a file of edgelists into a graph
    graph = readsGraphFromFile(filePath, nodeList)
    # extracts adjacency matrix from graph
    adjMatrix = getsAdjacencyFromGraph(graph)

    return adjMatrix;

def buildsSupermatrix(story):
    
    nt = len(story.graphs) # total number of time layers
    np = len(story.characters) # total number of characters
    timelayer = 0
    matrix = numpy.zeros((np*nt, np*nt))

    
    for graph in story.graphs:
        
        # extracts adjacency matrix from graph
        adjMatrix = getsAdjacencyFromGraph(graph)
        #print adjMatrix
        adjMatrix = numpy.multiply(story.epsilon, adjMatrix) # multiplies adjacency matrix by a given weight 'epsilon'

        ## allocates the adj matrix into the diagonal matrix at the appropriate position time-wise
        ## example: matrix[3 : 6][3 : 6] will receive the values of the 3x3 adj matrix of a story with 3 characters with data from the story's second chapter 
        matrix[np*timelayer : np*(timelayer+1), np*timelayer : np*(timelayer+1)] = adjMatrix
        timelayer += 1

    establishTemporalLinks(matrix, nt, np)
    
    return matrix;


def getsAdjacencyRemove1(filePath, fullNodeList, indexToBeRemoved):
          
    # read edgelist file into a graph
    graph = readsGraphFromFile(filePath, fullNodeList)
    
    # removes target node
    graph.delete_vertices(indexToBeRemoved)
            
    # extracts adjacency matrix from graph
    adjMatrix = getsAdjacencyFromGraph(graph)

    return adjMatrix;

def buildsSupermatrixRemove1(files, fullNodeList, epsilon, indexToBeRemoved):
    
    nt = len(files) # total number of time layers
    np = len(fullNodeList) - 1 # total number of characters minus the one being removed

    matrix = numpy.zeros( (np*nt, np*nt) )
    timelayer = 0
    
    for file in files:
        adjMatrix = getsAdjacencyRemove1(file, fullNodeList, indexToBeRemoved)

        #print '\nChapter', timeLayer+1
        #print '\nAdjacency matrix before epsilon:', adjMatrix
        
        adjMatrix = numpy.multiply(epsilon, adjMatrix) # multiplies adjacency matrix by a given weight 'epsilon'

        # atribui a matriz de adj. a matriz diagonal do tempo correspondente
        ## exemplo: as entradas de matriz[0 : 3][0 : 3] recebem os valores da matriz quadrada de adjacencia 3x3 de 3 personagens no tempo 1
        matrix[np*timelayer : np*(timelayer+1), np*timelayer : np*(timelayer+1)] = adjMatrix
        timelayer += 1

    establishTemporalLinks(matrix, nt, np)

    return matrix;



#not in use
def getsAllCharactersNcol(chapterList):

    characterList = []

    for file in chapterList:
        g = Graph.Read_Ncol(file, names=True, directed=False, weights=False)

        charactersInChapter = g.vs["name"]

        for name in charactersInChapter:
                if name not in characterList:
                        characterList.append(name)

    characterList.sort() # sorts list alphabetically (NOT alphanumerically!)

    return characterList;


def getsCharactersEdgesWeights(story):

    nf = len(story.chapterfiles)
    for f, file in enumerate(story.chapterfiles):
        edgelist = []
        weightlist = []
        for i, line in enumerate(open(file, 'r', encoding='utf-8')):
            original = line
            if not (re.match('\s*\r?\n', line)):
                line = line.split('#', 1)[0]
                if not (re.match('^\s*$', line)):
                    try:
                        elements = line.split()
                        n = len(elements)
                        if n < 2:
                            print '\nError: line segment ("%s") has too few elements. Edge format: Node1 Node2 (weight)' % line.split('\n', 1)[0]
                            print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )
                            sys.exit()
                            
                        elif n > 3: 
                            print '\nError: line segment ("%s") has too many elements. Edge format: Node1 Node2 (weight)' % line.split('\n', 1)[0]
                            print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )
                            sys.exit()
                            
                        elif n==3:
                            p1, p2, weight = elements
                            try:
                                float(weight) # tests whether third element of line is a number

                                if re.match('([+,-])?([I,i][N,n][F,f]|[N,n][A,n][N,n])$', weight):  # tests for inf and nan
                                    raise ValueError
                                
                                if p1 == p2:
                                    print "\nWARNING: loop detected"
                                    print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )

                                index1 = getsCharacterIndex(p1, story.characters)
                                index2 = getsCharacterIndex(p2, story.characters)
  #                              addsOccurrence(index1, story.characterOccurrences, f, nf)
#                                addsOccurrence(index2, story.characterOccurrences, f, nf)

                                edgelist.append( (index1, index2) )
                                weightlist.append( float(weight) )
                                        
                            except ValueError:
                                print '\nError: third element (edge weight) is not a number'
                                print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )
                                sys.exit()
                                
                        else: #n==2
                            p1, p2 = elements
                            if p1 == p2:
                                print "\nWARNING: loop detected"
                                print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )

                            index1 = getsCharacterIndex(p1, story.characters)
                            index2 = getsCharacterIndex(p2, story.characters)
 #                           addsOccurrence(index1, story.characterOccurrences, f, nf)
#                            addsOccurrence(index2, story.characterOccurrences, f, nf)

                            edgelist.append( (index1, index2) )
                            weightlist.append(1.0)  # default weight of 1.0

                    except ValueError:
                        print '\nError splitting this line segment: "%s"'%line.split('\n', 1)[0]
                        print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )
                        sys.exit()  

        story.edgelists.append(edgelist)
        story.weightlists.append(weightlist)

    return;

               
def eigenvectorSubroutine(story):

    autovetor, maiorAutovalor, autovetores, autovalores = getsOrderedEigs(story.supermatrix)
    #print 'Maior autovalor:', maiorAutovalor, '\n'
    #print 'Autovetor associado a lambda =', maiorAutovalor,':\n', autovetor, '\n'


 #   numpy.savetxt("eigenvector.txt", autovetor)


    np = len(story.characters)
    nt = len(autovetor)/np

    # Multiplies eigenvector by -1 if some element is negative for clarity (the vector remains correct)
    
    i=0
    while abs(autovetor[i]) < .00005:
        i += 1

    if autovetor[i] < -.00005:
        autovetor = numpy.multiply(-1, autovetor)
        

    story.jointEigenvector = autovetor

    # Marginal Layer, Marginal Node, Conditional and Time Averaged Conditional Centralities:

    story.mlc, story.mnc = marginalCentralities(story.jointEigenvector, nt, np)
    #print "Conditional Centralities... ",

    story.conditionalEigenvector = conditionalCentralities(story.jointEigenvector, story.mlc, nt, np)
    #print "\x1b[2K\rConditional Centralities... Done."

    story.timeAvgArray = timeAverage(story.conditionalEigenvector, nt, np)

    #Maximum centrality in TAC array
    story.timeAvgMaxCentrality = numpy.amax(story.timeAvgArray)

    #Freeman Index
    story.freemanIndex = freemanCentralization(story.timeAvgArray, np, story.timeAvgMaxCentrality)
    
    return;


def getsAllVitalitiesAIO(story):
    np = len(story.characters)
    
    story.absvitalities = numpy.zeros(np)
    story.vitalities = numpy.zeros(np)

    substory = story_struct()
    substory.filepath = story.filepath
    substory.chapterfiles = story.chapterfiles
    substory.epsilon = story.epsilon

    freemanIndexesList = []

    for index in range(np):
        print "\x1b[2K\rVitalities...",index+1,"/",np,
        sys.stdout.flush()
        
        aioGraph = story.graphs[0].copy()
        
        aioGraph.delete_vertices(index)  # removes target node
        adjMatrix = getsAdjacencyFromGraph(aioGraph) # extracts adjacency matrix from graph
        substory.supermatrix = adjMatrix
        
        substory.characters = list(story.characters)
        removedCharacter = substory.characters.pop(index)
            
        eigenvectorSubroutine(substory)

        #print 'Without',removedCharacter,':'
        #printsArrayInterpretation(substory)

    
        value = substory.freemanIndex
        freemanIndexesList.append(value)

        value = value - story.freemanIndex
        story.vitalities[index] = value
        story.absvitalities[index] = abs(value)

    print "\x1b[2K\rVitalities... Done."
    sys.stdout.flush()
    
    story.freemanIndexWhenEachAbsent = numpy.array(freemanIndexesList)

    return;

    

def getsStoryArraysRemove1(substory, fullCharacterList, indexToBeRemoved):

    substory.supermatrix = buildsSupermatrixRemove1(substory.chapterfiles, fullCharacterList, substory.epsilon, indexToBeRemoved)

    substory.characters = list(fullCharacterList)
    removedCharacter = substory.characters.pop(indexToBeRemoved)
    
    eigenvectorSubroutine(substory)

    #print 'Without',removedCharacter,':'
    #printsArrayInterpretation(substory)

    return;

def getsAllVitalities(story):

    np = len(story.characters)

    story.absvitalities = numpy.zeros(np)
    story.vitalities = numpy.zeros(np)

    substory = story_struct()
    substory.filepath = story.filepath
    substory.chapterfiles = story.chapterfiles
    substory.epsilon = story.epsilon

    freemanIndexesList = []
    
    for index in range(np):
        print "\x1b[2K\rVitalities...",index+1,"/",np,
        
        sys.stdout.flush()
        getsStoryArraysRemove1(substory, story.characters, index)

        value = substory.freemanIndex
        freemanIndexesList.append(value)

        value = value - story.freemanIndex
        story.vitalities[index] = value
        story.absvitalities[index] = abs(value)

    print "\x1b[2K\rVitalities... Done."
    sys.stdout.flush()
    story.freemanIndexWhenEachAbsent = numpy.array(freemanIndexesList)

    return;

def printsVitalities(story):
    
    print 'Vitalities:\n'
    for p in range(np):
            print story.characters[p], ':',story.vitalities[p]

    return;
                
## no longer in use
def getsCharacters(filePath):
    g = Graph.Read_Ncol(filePath,names=True,directed=False,weights=True)

    ncharacters = g.vcount()

    characterList = g.vs["name"]

    return ncharacters, characterList;


# function getsAdjacencyMatrixHeader (currently not in use!)
def getsAdjacencyMatrixHeader(filePath, epsilon, ncharacters, timeLayer):

    # reads edgelist file as a graph with igraph, with arguments:
    # - path to a *.txt edgelist (WITH character header to keep the characters indexes fixed before eliminating the header and reading the edges!!)
    # - names=True interprets node labels as their names
    # - directed=False interprets graph as undirected
    # - weights=True and the third column are ignored by this particular function (Read_Ncol)
    g = Graph.Read_Ncol(filePath,names=True,directed=False,weights=True)

    g.es[ : ncharacters].delete() # deletes the header of #ncharacters loop edges

    adjMatrix = g.get_adjacency() # matrix received from igraph as a list of lists; requires subsequent rearrangement for correct use 

    adjMatrix = numpy.array(adjMatrix.data) # lists properly arranged into matrix form

    print '\nChapter', timeLayer+1
    #print '\Adjacency matrix:', adjMatrix

    adjMatrix = numpy.multiply(epsilon, adjMatrix) # multiplies adjacency matrix by a given weight 'epsilon'
    
    #print '\Adjacency matrix w/ epsilon:', adjMatrix

    print adjMatrix;
    return adjMatrix;


def establishTemporalLinks(matrix, nlayers, ncharacters):

    for t in range(nlayers - 1): # [0, nlayers-1) b/c there are only nlayers-1 temporal links in each triangular
        for p in range(ncharacters):
            i = ncharacters * t + p
            j = ncharacters * (t + 1) + p
            matrix[i][j] = matrix[j][i] = 1
    return;


def getsCharacterIndex(name, characterList):

    lowerName = name.lower()
    
    try:
        i = next(i for i,character in enumerate(characterList) if character.lower() == lowerName)
        
    except StopIteration:
        i = len(characterList)
        characterList.append(name)

    return i;


def getsCharacterIndexCaseSensitive(name, characterList):


    try:
            i = characterList.index(name)
            
    except:
            i = len(characterList)
 #           checksSimilarity(name, characterList)  # maybe someday :(
            characterList.append(name)

    return i;


# calculates degree without building an iGraph graph.
def addsOccurrence(index, characterOccurrences, chapter, nchapters):

    try:
        characterOccurrences[index][chapter] += 1
            
    except IndexError:
        characterOccurrences.append(numpy.zeros((nchapters), dtype=numpy.int))
        characterOccurrences[index][chapter] += 1
        
    return; 
    

def getsEigenvector(matrix):

    eigenvalues, eigenvectors = numpy.linalg.eig(matrix)

    eigIndexes_decreasingOrder = eigenvalues.argsort()[::-1] # sorts the *indexes* of eigenvalues array
                                                             # by corresponding value from largest
                                                             # to smallest eigenvalue (decreasing order).
                                                             # e.g.: output [1,0,4,3] means that
                                                             # eigenvalues[1] holds the largest eigval,
                                                             # and eigenvalues[3] holds the smallest.

    index = eigIndexes_decreasingOrder[0] # index of the largest eigenvalue and largest corresponding
                                          # eigenvector column given by the first element of decreasingly
                                          # ordered (eigenvalue-wise) list of indexes. 
    
    largestEigenvalue = eigenvalues[index]
    largestEigenvector = eigenvectors[ : , index] # gets the matrix column corresponding to the index, sweeping across all matrix rows


    if largestEigenvector[0] < 0:                                       # if first element negative,
            largestEigenvector = numpy.multiply(-1, largestEigenvector) # multiplies entire eigenvector
                                                                        # by -1 for convenience. 
    
    return largestEigenvector.real, largestEigenvalue.real;


def getsOrderedEigs(matrix):

    eigenvalues, eigenvectors = numpy.linalg.eig(matrix)
    
    eigIndexes_decreasingOrder = eigenvalues.argsort()[::-1] # sorts the *indexes* of eigenvalues array
                                                             # by corresponding value from largest
                                                             # to smallest eigenvalue (decreasing order).
                                                             # e.g.: output [1,0,4,3] means that
                                                             # eigenvalues[1] holds the largest eigval,
                                                             # and eigenvalues[3] holds the smallest.

    eigenvalues = eigenvalues[eigIndexes_decreasingOrder] # reorders eigenvalues largest-to-smallest
                                                          # according to the list of indexes

    eigenvectors = eigenvectors[ : , eigIndexes_decreasingOrder ] # reorders eigenvector *columns* in the matrix
                                                                  # in the order of their corresponding eigvalues

    largestEigenvalue = eigenvalues[ 0 ]
    largestEigenvector = eigenvectors[ : , 0 ]

    if largestEigenvector[0] < 0:                                       # if first element negative,
            largestEigenvector = numpy.multiply(-1, largestEigenvector) # multiplies all eigenvectors
            eigenvectors = numpy.multiply(-1, eigenvectors)             # by -1 for clarity. 

    return largestEigenvector.real, largestEigenvalue.real, eigenvectors.real, eigenvalues.real; 

def marginalCentralities(centralityEigenvector, nlayers, nnodes):

    # In case arrays were not initialized zeroed-out: 
    layerArray = numpy.zeros(nlayers, dtype='float64')
    nodeArray = numpy.zeros(nnodes, dtype='float64')
    
    for t in range(nlayers):
        for p in range(nnodes):
            i = t*nnodes + p
            layerArray[t] += float(centralityEigenvector[i])
            nodeArray[p] += float(centralityEigenvector[i])
    return layerArray, nodeArray;

def conditionalCentralities(centralityEigenvector, layerArray, nlayers, nnodes):

    conditionalArray = numpy.empty(nlayers*nnodes, dtype='float64') 
    numpy.copyto(conditionalArray, centralityEigenvector, casting='unsafe')

    for t in range(nlayers):
        for p in range(nnodes):
                i = t*nnodes + p
                conditionalArray[i] /= float(layerArray[t])
    return conditionalArray;

def timeAverage(originalArray, nlayers, nnodes):

    # Makes sure new array is zeroed out 
    averagedArray = numpy.zeros(nnodes)
    
    for p in range(nnodes):
            for t in range(nlayers):
                    i = t * nnodes + p
                    averagedArray[p] += originalArray[i]

            averagedArray[p] /= nlayers

    return averagedArray;
    

def freemanCentralization(array, np, maxcentrality):

    sigma = 0 #summation
    normalizer = math.sqrt(np*(np-1)) # normalization factor 
    denominator = (np-1)*(np-2)
    for p in range(np):
            sigma += (maxcentrality - array[p])
    sigma /= denominator
    sigma *= normalizer
    return sigma;


def getsEVCents(graphs, weightlists):

    if len(graphs) != len(weightlists):
        print "Error (getsEVCents): number of graphs doesn't match number of weightlists"
        sys.exit()
    else:
        evclists = []
        for i, graph in enumerate(graphs):
            evclists.append(graph.evcent(directed=False, scale=False, weights=weightlists[i], return_eigenvalue=False) )

        return evclists;

# not in use
def new_getsEVCents(chapters):

    evclists = []
    for i, chapter in enumerate(chapters):
        evlist = chapter.graph.evcent(directed=False, scale=False, weights=chapter.weightlist, return_eigenvalue=False)
        chapter.evcents = evlist
        evclists.append(evlist)

    return evclists;

# "Culling" of unimportant characters - by cumulative density or sheer numbers
def cull(array, watershedPercentage, maxN):

    indexes = argsortDescending(array)

    watershed = watershedPercentage*(numpy.sum(array))
    cumulative = 0
    cullpoint = 0
    
    while cumulative < watershed and cullpoint < maxN-1 :
        cumulative += array[indexes[cullpoint]]
        cullpoint += 1

    cullpoint += 1
    indexes = indexes[:cullpoint]

    return indexes;

        

def printsArrayInterpretation(story):

    # Interpretation and visualization of the resulting arrays:

    np = len(story.characters)

    nt = len(story.jointEigenvector)/np
    
    maxlength = 0
    for p in range(np):
        maxlength = max(maxlength, len(story.characters[p]))
    
    if np*nt > 200:
        print "Joint and Conditional centralities omitted (" + str(np*nt) + " values each)."
        instr = str(raw_input("Print anyway? ('Y'/'y': yes; any other input: no)\n"))

    if np*nt <= 200 or instr == 'y' or instr == 'Y':
        print "\nJoint centrality, epsilon = "+str(story.epsilon)+':\n'
        for t in range(nt):
            print '' # line break between time layers
            for p in range(np):
                    # a time layer, a character's name and array index,
                    # and its corresponding joint and conditional centralities:
                    print 'T =',story.range[0]+t,'|', str('%-'+str(maxlength)+'s') % story.characters[p], ':', story.jointEigenvector[t*np + p]
        print "\n------------------------------------------------"
        print "\n\nConditional centrality, epsilon = "+str(story.epsilon)+':\n'
        for t in range(nt):
            for p in range(np):
                print 'T =',story.range[0]+t,'|',str('%-'+str(maxlength)+'s') % story.characters[p], ':', story.conditionalEigenvector[t*np + p]
        print "------------------------------------------------"
    print "\nMarginal Layer Centrality (MLC), epsilon = "+str(story.epsilon)+":\n"
    for t in range(nt):
        print str('T = %-'+ str(len(str(nt))) +'s') % str(story.range[0]+t), '|', story.mlc[t]
    print "\n------------------------------------------------"
    print '\nMarginal Node Centralities (MNC) for chapter range',str(story.range) +':\n'
    descending = argsortDescending(story.mnc)
    for p in descending:
        print str('%-'+str(maxlength)+'s') % story.characters[p], ':', story.mnc[p]

            
    #print 'Time averaged conditional (TAC) array:\n', story.timeAvgArray
#    print '\nMaximum centrality in TAC arrayfor chapter range',str(story.range) +':', story.timeAvgMaxCentrality
    print '\nFreeman Index of story for chapter range',str(story.range) +':', story.freemanIndex

    print "------------------------------------------------"
    
    return;


#includes joint+cond centralities
def createsComparisonFile(story, chapter_idx, indexes, savename):

        graph = story.graphs[chapter_idx]

        np = len(story.characters)
        
        if not os.path.exists(story.commonFolder):
            os.makedirs(story.commonFolder)        
        
        newfile = open(savename, "w+", encoding='utf-8')
        
        activeChars = 0
        for p in indexes:
            if graph.degree(p) != 0:
                activeChars += 1

        if story.jointEigenvector != []:
            
            newtext = [unicode('%-22s'%"Character" + ' %-10s'%"Degree" + ' %-20s'%"EV Centrality"+ ' %-20s'%"Betweenness" + ' %-25s'%"Normalized Betweenness" + ' %-20s'%"Closeness" + ' %-20s'%"Joint Centrality"+' %-20s'%"Conditional Centr."+ "\n")]
        
            for p in indexes:
                newtext.append(u'\n')
                newtext.append(u'%-22s '% story.characters[p] )
                newtext.append(u'%-10s '% graph.degree(p) )
                newtext.append(u'%-20s '% story.evcentlists[chapter_idx][p] )
                newtext.append(u'%-20s '% graph.betweenness(p) )
                newtext.append(u'%-25s '% graph.betweenness(p) )
                newtext.append(u'%-20s '% graph.closeness(p) )
 ###               newtext.append(u'%-20s '% story.chapters[chapter_idx].evcents[p] )
                newtext.append(u'%-20s '% story.jointEigenvector[chapter_idx*np + p] )
                newtext.append(u'%-20s '% story.conditionalEigenvector[chapter_idx*np + p] )

        else: # no joint/cond cent
            
            newtext = [unicode('%-22s'%"Character" + ' %-10s'%"Degree" + ' %-20s'%"EV Centrality"+ ' %-20s'%"Betweenness" + ' %-25s'%"Normalized Betweenness"+ ' %-20s'%"Closeness"  +"\n")]
        
            for p in indexes:
                newtext.append(u'\n')
                newtext.append(u'%-22s '% story.characters[p] )
                newtext.append(u'%-10s '% graph.degree(p) )
                newtext.append(u'%-20s '% story.evcentlists[chapter_idx][p] )
                newtext.append(u'%-20s '% graph.betweenness(p) )
                newtext.append(u'%-25s '% graph.betweenness(p) )
                newtext.append(u'%-20s '% graph.closeness(p) )

        if story.plottedIdx != []:

            newtext.append(u'\n\n\nFocus characters: ' + u', '.join(char.capitalize() for char in (story.characters[p] for p in story.plottedIdx)) + '\n')

            adj = getsAdjacencyFromGraph(graph)

            focus_lines = []
            focus_degrees = []
            choosetwo = itertools.combinations_with_replacement(story.plottedIdx, 2)

            
            for tuple in choosetwo:
                p1 = tuple[0]
                p2 = tuple[1]
                deg = adj[p1][p2]
                focus_degrees.append(deg)

                if deg > 1:
                    focus_lines.append(u'\n%s and %s interacted %d times.'%(story.characters[p1].capitalize(), story.characters[p2].capitalize(), deg))

                elif deg == 1:
                    focus_lines.append(u'\n%s and %s interacted once.'%(story.characters[p1].capitalize(), story.characters[p2].capitalize()))

                elif deg == 0:
                    focus_lines.append(u'\n%s and %s did not interact.'%(story.characters[p1].capitalize(), story.characters[p2].capitalize()))


            for i in argsortDescending(numpy.array(focus_degrees)):
                newtext.append( focus_lines[i] )
                

            # these commands below can be moved to a proper function and this function can be called here (as well as in other situations but in this case, one probably wants a printout in the screen hence the function must also receive a parameter indicating whether it is to be printed in this file or in the screen and then use print for the screen
            newtext.append(u'\n\ndensity of graph: ')
            newtext.append(u'%-20s '% graph.density() )
            newtext.append(u'\ndiameter of graph: ')
            newtext.append(u'%-20s '% graph.diameter() )
            newtext.append(u'\nAvg. transitivity of graph: ')
            newtext.append(u'%-20s '% graph.transitivity_avglocal_undirected(mode="zero") ) #  mode=nan produces mainly same transitivity
            newtext.append(u'\nDegree distr. of graph: ')
            newtext.append(u'%-20s '% graph.degree_distribution(bin_width=10) )  # o par. bin_width tem que ser passado pelo usuario pois depende da qtdd. de caps o grau varia

        newfile.writelines(newtext)
        newfile.close()

        return;



def printsComparison(story, chapter_idx, graph, indexes):

        np = len(story.characters)
       
        print '\n\n%-22s'%"Character",
        print ' %-10s'%"Degree",
        print ' %-20s'%"EV Centr.",
        print ' %-20s'%"Betweenness",
        print ' %-20s'%"Closeness",
        if story.jointEigenvector != []:
            print ' %-20s'%"Joint Centr.",
            print ' %-20s'%"Conditional Centr."
        else:
            print ""
        
        for p in indexes:
            print u'%-22s '% story.characters[p],
            print '%-10s '% graph.degree(p),
            print '%-20s '% story.evcentlists[chapter_idx][p],
            print '%-20s '% graph.betweenness(p),
            print '%-20s '% graph.closeness(p),
            if story.jointEigenvector != []:
                print '%-20s '% story.jointEigenvector[chapter_idx*np + p],
                print '%-20s '% story.conditionalEigenvector[chapter_idx*np + p]
            else:
                print ""

        if story.plottedIdx != []:

            print (u'\n\n\nFocus characters: ' + u', '.join(char.capitalize() for char in (story.characters[p] for p in story.plottedIdx)) + '\n')

            adj = getsAdjacencyFromGraph(graph)

            focus_lines = []
            focus_degrees = []
            choosetwo = itertools.combinations_with_replacement(story.plottedIdx, 2)

            
            for tuple in choosetwo:
                p1 = tuple[0]
                p2 = tuple[1]
                deg = adj[p1][p2]
                focus_degrees.append(deg)

                if deg > 1:
                    focus_lines.append(u'%s and %s interacted %d times.'%(story.characters[p1].capitalize(), story.characters[p2].capitalize(), deg))

                elif deg == 1:
                    focus_lines.append(u'%s and %s interacted once.'%(story.characters[p1].capitalize(), story.characters[p2].capitalize()))

                elif deg == 0:
                    focus_lines.append(u'%s and %s did not interact.'%(story.characters[p1].capitalize(), story.characters[p2].capitalize()))


            for i in argsortDescending(numpy.array(focus_degrees)):
                print ( focus_lines[i] )

        


        return;

    
def drawsLineplot(title, y_axis, x_axis, xtickslabels, y_label, x_label, linelabels, savename, nfeatured): 

    nlabels = len(linelabels)
    for p in range(nlabels):
        values = y_axis[p, :]
        
        if (p < nfeatured):
            plt.plot(x_axis, values, '-', label=linelabels[p], linewidth=2)
            
        else:
            plt.plot(x_axis, values, '-', linewidth=1)

    plt.title(title, fontsize =12, fontweight='demibold')
    ax = plt.gca() # grabs the current axis
 #   ax.set_ylim([0, .20]) ##fix y axis range
    plt.ylabel(y_label, fontsize=10)
    plt.xlabel(x_label, fontsize=10)
    plt.xticks(x_axis)
    ax.set_xticklabels(xtickslabels, fontsize = 7) # sets the labels to display at x ticks
    plt.legend()
    if not os.path.exists(story.commonFolder):
        os.makedirs(story.commonFolder)
    plt.savefig(savename, bbox_inches = 'tight')
    plt.show()
    return;

    
def plotsVitalitiesLineGraph(title, y_axis, names, freemanIndex, distinguished, savename):

    np = len(names)

    plt.plot(range(1, np+1), y_axis, '-', linewidth = 1.5)
    
    plt.axhline(y=freemanIndex, color= 'red')

    colors = ['red', 'black', 'magenta', 'lime', 'orange']

    if np <= 5:
        xticks = range(1, np+1)
        for idx in range(np):
            plt.plot(idx+1, y_axis[idx], marker='o', markersize=4, color = colors[idx], label=names[idx])

    else:
        xticks = [1]
        cont = 0
        for idx in range(np):
            if idx in distinguished:
                plt.plot(idx+1, y_axis[idx], marker='o', markersize=4, color = colors[cont], label=names[idx])
                xticks.append(idx+1)
                cont+=1
            else:
                plt.plot(idx+1, y_axis[idx], marker='o', markersize = 2, color='blue')
        if np not in distinguished:
            xticks.append(np)

    ax = plt.gca() # grabs the current axis
    #ax.set_ylim([.125, .155]) #same limit as in the paper for the Alice case
    ax.set_xticks(xticks) # chooses which x locations to have ticks
    ax.set_xticklabels(xticks, fontsize = 7) # sets the labels to display at those ticks

    plt.title(title, fontsize =12, fontweight='demibold')
 
    plt.ylabel('Freeman Index', fontsize=10)
    plt.xlabel('Character', fontsize=10)
    plt.legend()
    if not os.path.exists(story.commonFolder):
        os.makedirs(story.commonFolder)
    plt.savefig(savename, bbox_inches = 'tight')

    plt.show()

    return;

def plotsHeatMap(titleString, arrayformap, names, nchapters, minimal, maximal,filename):
        
        pcolor(arrayformap ,cmap='hot',vmin=minimal, vmax=maximal) #'YlGnBu')  # 'PuOr') # 'PuRd') #'hot')  #'Reds') #'Set1')  #cmap='Accent') #map='GnBu'
        colorbar()
        
        title(titleString)
        yticks( range(len(names)), names, fontsize=8)
 #       plt.ylabel('Characters', fontsize=8)
        plt.xlabel('Chapters', fontsize=8)
        xticks(range(1,nchapters+1))
        if not os.path.exists(story.commonFolder):
            os.makedirs(story.commonFolder)
        plt.savefig(filename, bbox_inches = 'tight')
        plt.show()

        return;

def drawsBarplot(title, data, names, filename):

        ticks = numpy.arange(len(names))
        total = len(names)
    
        fig, ax = plt.subplots() 
        ax.barh(ticks[::-1], data[::-1], color=["purple", "blue"])
        ax.set_yticks(ticks[::-1])
        ax.set_yticklabels(names[::-1], minor=False, fontproperties=font_manager.FontProperties( size=8))
        for i, v in enumerate(data[::-1]):
            if (isinstance(v, int)):
                labelv=str(v)
            else:
                labelv=str('%.2f' % v)
            ax.text(v, total -1 -i , labelv, color='black')
            
        plt.title(title)
        plt.ylabel("Characters", fontproperties=font_manager.FontProperties( size=10))

        fig.set_size_inches(5., 12.5)
        if not os.path.exists(story.commonFolder):
            os.makedirs(story.commonFolder)

        fig.savefig(filename,bbox_inches='tight')

        return;



def critSum(story, method):

    if method not in ['d', 'c', 'b', 'mnc']:
        raise Exception("Method %s not implemented" % method_name)

    else:
        np = len(story.characters)
        nt = len(story.graphs)
        sumArray = numpy.zeros((np))
        if method == 'mnc':
            sumArray = story.mnc.copy()
        elif method == 'd':       
            for t in range(nt):
                for p in range(np):
                    sumArray[p] += story.graphs[t].degree(p)
        elif method == 'c':       
            for t in range(nt):
                for p in range(np):
                    sumArray[p] += story.graphs[t].closeness(p)
        elif method == 'b':       
            for t in range(nt):
                for p in range(np):
                    sumArray[p] += story.graphs[t].betweenness(p)

        return sumArray;


# maybe someday :(
def checksSimilarity(name, characters):

    for character in characters:
        similarity = difflib.SequenceMatcher(None, name, character)
        ratio = similarity.ratio()
        if ratio > 0.70:
            print '"%s" and "%s", similarity ratio: %f'%(name,character, ratio)

    return;
def similarNames(story):

    np = len(story.characters)

    matrix = numpy.ones((np, np))

    for i in range(np):
        for j in range(np):
            if (i != j) and matrix[i][j] == 1:
                nameSimilarity = difflib.SequenceMatcher(None, story.characters[i], story.characters[j])                                          
                matrix[i][j] = matrix[j][i] = ratio = nameSimilarity.ratio()
                if ratio > 0.70:
                    print '\n\nSimilarity ratio: %f'%(ratio)
                    print u'"%-22s": '%story.characters[i],
                    for n in story.characterOccurrences[i]:
                        print '{:<3}'.format(n),
                    print u'\n"%-22s": '%story.characters[j],
                    for m in story.characterOccurrences[j]:
                        print '{:<3}'.format(m),
    return; 



def readsEdgeLine(file, line, i, chapter, characters):

    original = line
    line = line.split('#', 1)[0]    # split on '#' if existant
    if not (re.match('^\s*$', line)):   # if remnant is not just emptyspace (left of commented segment)
        
        try:
            elements = line.split()
            n = len(elements)
            if n < 2:
                print '\nError: line segment ("%s") has too few elements. Edge format: Node1 Node2 (weight)' % line.split('\n', 1)[0]
                print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )
                sys.exit()
            
            elif n > 3: 
                print '\nError: line segment ("%s") has too many elements. Edge format: Node1 Node2 (weight)' % line.split('\n', 1)[0]
                print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )
                sys.exit()
            
            elif n==3:
                p1, p2, weight = elements
                try:
                    float(weight) # tests whether third element of line is a number

                    if re.match('([+,-])?([I,i][N,n][F,f]|[N,n][A,n][N,n])$', weight):  # tests for inf and nan
                        raise ValueError
                    
                    if p1 == p2:
                        print "\nWARNING: loop detected"
                        print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )

                    index1 = getsCharacterIndex(p1, characters)
                    index2 = getsCharacterIndex(p2, characters)

                    chapter.edgelist.append( (index1, index2) )
                    chapter.weightlist.append( float(weight) )
                        
                except ValueError:
                    print '\nError: third element (edge weight) is not a number'
                    print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )
                    sys.exit()
                
            else: #n==2
                p1, p2 = elements
                if p1 == p2:
                    print "\nWARNING: loop detected"
                    print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )

                index1 = getsCharacterIndex(p1, story.characters)
                index2 = getsCharacterIndex(p2, story.characters)

                chapter.edgelist.append( (index1, index2) )
                chapter.weightlist.append(1.0)  # default weight of 1.0

        except ValueError:
            print '\nError splitting this line segment: "%s"'%line.split('\n', 1)[0]
            print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, original.split('\n', 1)[0] )
            sys.exit()

    return;


def readsChaptersWithSeparator(files, chapters, characters, pattern, keywords):

    original = pattern
    
    original = re.sub('\\\d\+', '(\d+)', original) # keeps original pattern while adding parentheses for the strict expression
    original = re.sub('\\\d\*', '(\d*)', original)
    original = re.sub('\\\d(?![+*])', '(\d)', original)

    pattern = re.sub('\(\\\d\+\)', '\d+', pattern) # strips the parentheses for ease of treatment for the permissive expression
    pattern = re.sub('\(\\\d\*\)', '\d*', pattern)
    pattern = re.sub('\(\\\d\)', '\d', pattern)
    
    pattern = re.sub('(\[[\d\,\-\?\s]+\][|*+\?]?)+', '\d+', pattern ) # replaces (repeating or not) single-digit expressions like [0-9] or [1, 3 ][ 3-9 ] or [1-3]|[8,9] with \d+

    pattern = re.sub('(\d+)[*+]?[?]?', '\d+', pattern ) 
    pattern = re.sub('(\d\*?\+?\|\d\+?\*?)', '\d+', pattern )
    
    pattern = re.sub("[(](\\\d[+*]?)+[)]", "\d+", pattern) # replaces consecutive instances of \d+ with a single one
    pattern = re.sub("(\\\d[+*\?]?)+", "\d+", pattern)
    pattern = re.sub('\(\?\!\\\d\+\)', "\d+", pattern)
    pattern = re.sub("(\\\d[+*\?]?)+", "(\d+)", pattern) # repeats the process one last time, while adding back parentheses for the permissive exp.
    

    print "\n\nStrict pattern:",original
    print "Permissive pattern:",pattern,"\n"

    ignore = True
    
    for file in files:
        k = 0           # in-file chapter counter
        f = open(file, 'r', encoding='utf-8')

        i=0             # line counter

        whitespace = re.compile('^\s*\r?\n')
        strict = re.compile('^\s*'+original+'\s*')
        permissive = re.compile('^\s*'+pattern+'\s*')

        line = unicode(f.readline())
        while line:
            if not whitespace.match(line): # if line is not just whitespace

                ## compares line against pattern
                stmatch = strict.match(line)
            
                ## checks result and whether the line includes the specified keywords (should there be any)
                if stmatch and (keywords == [] or keywordsInLine(keywords, line.lower()) ):  # new chapter!

                    match = permissive.match(line) # captures, in effect, the variable groups by using the permissive expression
                    ignore = False
                    current = f.tell()
                    if k > 0:
                        endtell = current-len(line)
                        chapter.endpoint = endtell

                    k += 1
                    newchapter = chapter_struct()  
                    newchapter.file = file
                    newchapter.startpoint = current     ## gets starting byte position for chapter within file 

                    try:
                        newchapter.match = match.group(0).split('\n', 1)[0]   ## gets string that matched the expression
                        newchapter.regroups = [int(x) for x in match.groups() if x != None and x.isdigit() ]

                    
                    except AttributeError: # something went wrong while capturing the groups - reverts back to strict match
                        print "WARNING: group capture regex yielded no match. Using strict regex match instead."
                        print '   File: %s\n   Line %d: "%s"\n\n'%(file, i+1, line.split('\n', 1)[0] )

                        newchapter.match = stmatch.group(0).split('\n', 1)[0]                        
                        newchapter.regroups = [int(x) for x in stmatch.groups() if x != None and x.isdigit() ]


                    newchapter.match = newchapter.match.replace(" ","")
                    newchapter.match = "".join([x if (x.isalnum() or x in ".-_") else "_" for x in newchapter.match])
                    newchapter.match = newchapter.match.replace("..",".")
                    newchapter.headerline = line.split('\n', 1)[0]

                    print 'New match '+'('+ ",".join(str(x) for x in newchapter.regroups)+') in "%s"'%newchapter.headerline

                    story.chapters.append(newchapter)   ## adds new chapter to collection 

                    chapter = newchapter     # 'new' becomes 'current' (for edge-reading)

                elif permissive.match(line):  # if found variant of the regex outside the explicit range of this story
                    ignore = True         # this chapter is not in the story, so all subsequent edgelines are ignored.

                elif not ignore:
                    readsEdgeLine(file, line, i, chapter, characters)

            line = f.readline() # iterates while-loop
            i += 1 # line counter

        if k > 0: #found at least one chapter
            chapter.endpoint = os.path.getsize(file)    # final byte position for last chapter in file.
            
        f.close()
    return;


def readsEdgelistsWithBoundaries(chapters, characters):

    k=0
    for chapter in chapters:
        k += 1
        f = open(chapter.file, 'r', encoding='utf-8')
        f.seek(chapter.startpoint)

        bytes = chapter.endpoint - chapter.startpoint -1
        i=0
        print "============== k = %d =============="%k
        while bytes > 0:
            line = f.readline()
            i += 1
            bytes = bytes - len(line)
            print line

            readsEdgeLine(file, line, i, chapter, characters)

        f.close()


    return; 
                        
def readsSingleFile(story, file, separator):

    f = open(file, 'r', encoding='utf-8')

    line = f.readline()
    i = 0
    while line:
        original = line
        if not (re.match('\s*\r?\n', line)):
            line = line.split('#', 1)[0]
            if not (re.match('^\s*$', line)):   # if not just emptyspace left of comment
    
                regex = re.match('^\s*'+separator, line)
                if regex:  # new chapter
                    
                    newchapter = chapter_struct()
                    newchapter.file = file
                    print f.tell()
                    newchapter.ftell = f.tell()     ## gets starting byte position for chapter within file 
                    newchapter.title = regex.group(0)  ## gets string that matched the expression
                    newchapter.regroups = [ int(regex.group(i+1)) for i in range(len(regex.groups() ))]
                    

                    print "regroups! %d %d"%(newchapter.regroups[0], newchapter.regroups[1])
                    story.chapters.append(newchapter)   ## adds new chapter to collection 
                    chapter = newchapter

                else:
                    readsEdgeLine(file, line, i, chapter, story.characters)

                    
        line = f.readline() # iterates while-loop
        i += 1 # line counter

    print "no of chars: ", len(story.characters)

    f.close()
    return;


def getsCharacterListFromFile(story, file):
    
    with open(file, 'r', encoding='utf-8') as f:

        for line in f:
            for name in line.split():

                lowerName = name.lower()
                if lowerName in [ x.lower() for x in story.characters ]:
                    if lowerName not in [x.lower() for x in story.plottedCharacters ]:
                        i = getsCharacterIndex(name, story.characters)
                        story.plottedCharacters.append(story.characters[i])
                        story.plottedIdx.append(i)
                        
                else:
                    print 'WARNING: character "%s" in file "%s" not in story\'s list of characters.'%(name, file)
        
    return; 
                

def keywordsInLine(keywords, line):

    listedwords = re.findall(r'\w+', line, re.UNICODE)
    
    for keyword in keywords:
        if keyword in listedwords:
            return True

    return False;


def interpolatesColors(colorlist, nc): # using iGraph, interpolates a given list of colors into a new list with cardinality nc

    if len(colorlist) == 2:

        palette = GradientPalette(colorlist[0], colorlist[1], n=nc)

    else:

        palette = AdvancedGradientPalette(colorlist, n=nc)
    
    colors = [ palette.get(x) for x in range(nc) ]


    return colors;

def mostImportants(listaOriginal): #cria uma lista com o indice dos personagens mais importantes
    lista = listaOriginal
    top = [] #cria lista vaiz
    for i in range(15):
        maiorValor = max(lista)
        top.append(lista.index(maiorValor))
        lista[lista.index(maiorValor)] = 1
    return top


def indexmin(criterionArray, i1, i2):

    try:
            
        if criterionArray[i1] < criterionArray[i2]:
            return i1
        else:
            return i2

    except TypeError:

        if criterionArray(i1) < criterionArray(i2):
            return i1
        else:
            return i2

#================== Misc ===================#

# special characters: epsilon ε

# FOR TESTING: Freeman indexes when each character absent for Anos Rebeldes
 # [0.145854647391, 0.140395386089, 0.140450030666, 0.140333531608, 0.140929722842, 0.141116028683, 0.139998271826, 0.140188731004, 0.141197253769, 0.140436001896, 0.141332458313, 0.14004359927, 0.140340143924, 0.140854744701, 0.139979137375, 0.146632990521, 0.140140586141, 0.139973920753, 0.141784435619, 0.143478295755, 0.142757629717, 0.144507400229, 0.141440142757, 0.140066753769, 0.14003678828, 0.146946883724, 0.140197609777, 0.140666856092, 0.143515299617, 0.14993680662, 0.140169195969, 0.140003022542, 0.14011680956, 0.120861493157, 0.13998222897, 0.140497340283, 0.141952839226, 0.141216418679, 0.144304227623, 0.140419801396, 0.139974427308, 0.146962949235, 0.139985277653, 0.101517859526, 0.13999046858, 0.139974950285, 0.140076215567, 0.141792077172, 0.140282246599, 0.140600421311, 0.139973533741, 0.143566418972, 0.140123717963, 0.141197538003, 0.142691851475, 0.140898213459, 0.142151875483, 0.140121518478, 0.144795933559, 0.141420244928, 0.144740465923, 0.144620448028, 0.140385043055, 0.140428055773, 0.140963366576, 0.140760393435, 0.140637484999, 0.141313960387, 0.142120696537, 0.140158004267, 0.14261666928, 0.140193136272, 0.140303151978, 0.140007595934, 0.140369926303, 0.140005486259, 0.14035041103, 0.140052563797]


#================== Argparse and Validation ===================#

a = p.parse_args()


story = story_struct()
story.title = a.title
story.epsilon = a.epsilon

if (a.filename and a.singlefile):
    print "Error: -f and -sf are mutually exclusive!"
    sys.exit()

if not (a.filename or a.singlefile):
    print "Error: -f xor -sf required"
    sys.exit()


if a.keywords:
    story.keywords = [ word.lower() for word in a.keywords]
    print "\nKeywords: "+', '.join(story.keywords)
else:
    story.keywords = []



if (a.filename):
    if not (a.firstchapter and a.lastchapter):
        print "Missing either/both first file (-from) or last file (-to) arguments (e.g.: -from 5 -to 8)"
        sys.exit()        


    if not (validatesChapterRange(a.firstchapter, a.lastchapter)):
        print "Error: invalid chapter range "
        sys.exit()

        
    chapters = getsFilepaths(os.path.abspath(a.filename), a.firstchapter, a.lastchapter)


    story.filepath = os.path.abspath(a.filename)
    story.chapterfiles = chapters

    story.commonName = os.path.splitext(os.path.basename(a.filename))[0]

    story.commonFolder = os.path.abspath(os.path.dirname(a.filename))
    story.commonFolder = os.path.join(story.commonFolder, '')
    story.commonFolder += "Plots"

    story.commonPath = os.path.join(story.commonFolder, story.commonName)


    if a.title:
        story.title = a.title
    else:
        story.title = story.commonName

    story.range = [a.firstchapter, a.lastchapter]


        
    story.savestring = story.commonName +str(a.firstchapter)+"-"+str(a.lastchapter) 

    if a.lastchapter==a.firstchapter:
        story.rangestring = str(a.firstchapter)
    else:
        story.rangestring = str(a.firstchapter)+"-"+str(a.lastchapter)

    getsCharactersEdgesWeights(story)



elif (a.singlefile):

    if not (a.separator):
        print 'Missing separator string argument'
        sys.exit()

    elif (re.match('^\s*$', a.separator) ):
        print 'Invalid separator string.'
        sys.exit()
        
    elif not os.path.isfile(os.path.abspath(a.singlefile)):
        print 'File %s not found.'%(os.path.abspath(a.singlefile))
        sys.exit()


    else:
        story.filepath = os.path.abspath(os.path.dirname(a.singlefile))
        story.filepath = os.path.join(story.filepath, '')


        story.commonFolder = os.path.abspath(os.path.dirname(a.singlefile))
        story.commonFolder = os.path.join(story.commonFolder, '')
        story.commonFolder += "Plots"

        story.commonName = os.path.splitext(os.path.basename(a.singlefile))[0]

        story.commonPath = os.path.join(story.commonFolder, story.commonName)
        
        story.files = [os.path.abspath(a.singlefile)]
        
        readsChaptersWithSeparator(story.files, story.chapters, story.characters, a.separator, story.keywords)


        if len(story.chapters) == 0:
            print "No chapters found."
        elif len(story.chapters) == 1:
            story.savestring = story.commonName +'_'+ str(story.chapters[0].match)
            story.rangestring = str(story.chapters[0].match)
        else:
            story.savestring = story.commonName +'_'+ str(story.chapters[0].match)+'-'+str(story.chapters[-1].match)
            story.rangestring = str(story.chapters[0].match)+'-'+str(story.chapters[-1].match)

            
        if a.title:
            story.title = a.title
        else:
            story.title = story.commonName 

        np = len(story.characters)


        if a.keywords:
            if any(isinstance(element, list) for element in a.keywords): # checks if its a list of lists (multiple collections)
                for alist in a.keywords:
                    newcollection = collection_struct()
                    newcollection.keywords = [ word.lower() for word in alist]
                    story.collections.append(newcollection)

            else: # just one list of keywords
                newcollection = collection_struct()
                newcollection.keywords = [ word.lower() for word in a.keywords]
                story.collections.append(newcollection)
                                    
            
        for chapter in story.chapters:
            chapter.graph = makesGraphFromEdgesWeights(np, chapter.edgelist, chapter.weightlist)
            story.edgelists.append(chapter.edgelist)
            story.weightlists.append(chapter.weightlist)

            line = chapter.headerline.lower()            
            for collection in story.collections:
                if keywordsInLine(collection.keywords, line):
                    collection.chapters.append(chapter)
            
        

        
#=== Initializing structure from args ===#

numpy.set_printoptions(threshold=10000) # for unlimited: threshold='inf'

required = a.linePlot + a.heatMap + a.barPlot

if a.top:
    required.append(a.top)

if (a.joint or a.vitalities or a.tabj or a.firstOrderMoverScore or not(set(['jc','cc','mnc']).isdisjoint(required)) ):
    requiresSupermatrix = True
else:
    requiresSupermatrix = False

if (a.table or a.tabj or ('ec' in required)):
    requiresEVCent = True
else:
    requiresEVCent = False



np = len(story.characters)

#story.characterOccurrences = numpy.zeros((np, a.lastchapter-a.firstchapter+1))




watershedPercentage = a.percentageCutoff

print "\nSystem default encoding: " + sys.getdefaultencoding()

print "------------------------------------------------"
print 'Starring:', ', '.join(story.characters)
print '\n',np,'nodes across',

if a.filename:
    print a.lastchapter-a.firstchapter+1,'time layers.'
else:
    print len(story.chapters),'time layers.'
print "------------------------------------------------\n"


for collection in story.collections:
    if collection.chapters != []:
        print '\n',collection.keywords[0].capitalize()+' episodes ("'+'", "'.join(collection.keywords)+'"): '+ ", ".join(match for match in (chapter.match for chapter in collection.chapters))
        print ""
#    for p in range(np):
#        print p, '-', story.characters[p]

#================== Processes ===================#


if a.featuredCharactersFile or a.featuredCharactersList:

    if a.featuredCharactersFile:
        if not os.path.isfile(os.path.abspath(a.featuredCharactersFile)):
            print 'File %s not found.'%(os.path.abspath(a.featuredCharactersFile))
            sys.exit()
        else:
            getsCharacterListFromFile(story, os.path.abspath(a.featuredCharactersFile))

    if a.featuredCharactersList:

        for name in a.featuredCharactersList:
            lowerName = name.lower()
            
            if lowerName in [ x.lower() for x in story.characters ]:
                if lowerName not in [x.lower() for x in story.plottedCharacters ]:
                    
                    i = getsCharacterIndex(name, story.characters)
                    story.plottedCharacters.append(story.characters[i])
                    story.plottedIdx.append(i)
            else:
                print 'WARNING: character "%s" not in story\'s list of characters.'%(name)
        

    print "\nPlotted characters: "+ ("none." if story.plottedCharacters == [] else (", ".join(story.plottedCharacters)))



if (a.allInOne): #all chapters blended into one graph
    
    aioEdges = []
    aioWeights = []

    for elist in story.edgelists:
        aioEdges += elist
    for wlist in story.weightlists:
        aioWeights += wlist

        
    aioGraph = makesGraphFromEdgesWeights(len(story.characters), aioEdges, aioWeights)
    story.graphs = [aioGraph]

    if requiresSupermatrix:
        aioAdjMatrix = getsAdjacencyFromGraph(aioGraph)
        aioAdjMatrix = numpy.multiply(story.epsilon, aioAdjMatrix) # multiplies adjacency matrix by a given weight 'epsilon'

        story.supermatrix = aioAdjMatrix
        eigenvectorSubroutine(story)

    if requiresEVCent:
        story.evcentlists = getsEVCents([aioGraph], [aioWeights])

    if a.table or a.tabj:
        if a.table:
            endstring = "_comparisonTableAIO.txt"
        else: #tabj
            endstring = "_comparisonTableAIO_eps"+str(story.epsilon)+".txt"

        savename = story.commonPath + story.rangestring + endstring
            
        indexes = argsortDescending(numpy.array(aioGraph.degree()) )
        
        createsComparisonFile(story, 0, indexes, savename)
        print "Table file saved in "+savename
        
        printsComparison(story, 0, aioGraph, indexes)


if not a.allInOne: # each chapter read independently into separate graphs
    
    story.graphs = [] # clears list just in case (used with append)

    if (a.singlefile):
        for chapter in story.chapters:
            story.graphs.append(chapter.graph)
            
    elif (a.filename):
        total = len(story.edgelists)
        for t in range(total):
            print "\x1b[2K\rBuilding graphs...",t+1,"/",total,
            sys.stdout.flush()
            story.graphs.append( makesGraphFromEdgesWeights(len(story.characters), story.edgelists[t], story.weightlists[t]))
                
        print "\x1b[2K\rBuilding graphs... Done. \n",
        sys.stdout.flush()

    if requiresSupermatrix:
        story.supermatrix = buildsSupermatrix(story) # Supermatrix all set!
        eigenvectorSubroutine(story)

    if requiresEVCent:
        story.evcentlists = getsEVCents(story.graphs, story.weightlists)
        
    if a.table or a.tabj:
        if a.table:
            endstring = "_comparisonTable.txt"
        else:
            endstring = "_comparisonTable_eps"+str(story.epsilon)+".txt"
            
        total = len(story.graphs)
        if total > 0:
            for t, chapterGraph in enumerate(story.graphs):
                print "\x1b[2K\rMaking table files...",t+1,"/",total,
                sys.stdout.flush()


                savename = story.commonPath
                
                if a.filename:
                    savename += str(a.firstchapter+t) + endstring
                    
                elif a.singlefile:
                    savename += story.chapters[t].match + endstring

                indexes = argsortDescending(numpy.array(chapterGraph.degree()) )
                createsComparisonFile(story, t, indexes, savename)

                
            print "\x1b[2K\rMaking table files... Done. ",
            sys.stdout.flush()
            print "%d saved in %s"%(t+1,os.path.dirname(savename))

        
        
            
if a.joint:
        printsArrayInterpretation(story)


if a.firstOrderMoverScore:

    if story.epsilon == 0.0:
        print "Error: division by zero. Could not complete first-order Mover Score calculations."

    else:

        n_characters = len(story.characters)

        m_shape = story.supermatrix.shape

        n_layers = m_shape[0]/n_characters

        m_0 = numpy.zeros( m_shape )

        establishTemporalLinks(m_0, n_layers, n_characters)

     #   numpy.savetxt("mat.txt", m_0, fmt= '%d')

        eigvector_0, eigvalue_0 = getsEigenvector(m_0)

     #   numpy.savetxt("vzero.txt", eigvector_0)

        eigvector_1 = (eigvector_0 - story.jointEigenvector)/story.epsilon

        moverScore = numpy.zeros(n_characters)

        for p in range(n_characters):
            for t in range(n_layers):
                moverScore[p] += pow(eigvector_1[ n_characters*t + p], 2)

        moverScore = numpy.sqrt(moverScore)

        
        maxlength = 0
        for p in range(n_characters):
            maxlength = max(maxlength, len(story.characters[p]))

        print "\n- - - - - - - - - - - - - - - - - - - - - - - -"
        print "           First-order Mover Scores       "
        print "- - - - - - - - - - - - - - - - - - - - - - - -\n"

        for k,i in enumerate(argsortDescending(moverScore)):

            print '%3d'%(k+1)+'. ',str('%-'+str(maxlength)+'s') % story.characters[i], '|', '%20.8f' % moverScore[i]

        print "\n- - - - - - - - - - - - - - - - - - - - - - - -\n"




    

if a.vitalities or ('v' in a.linePlot):

    if a.singlefile:
        print "Warning: Vitality is not available with -sf yet!"
        a.vitalities = False
        if ('v' in a.linePlot):
            a.linePlot.remove('v')
        
    else:
        
        if a.allInOne:
            getsAllVitalitiesAIO(story)

        else:
            getsAllVitalities(story)

        if a.vitalities:
            printsVitalities(story)


if a.graphPlot:

    np = len(story.characters)
    total = len(story.graphs)
    t = 0

    for graph in story.graphs:

        print "\x1b[2K\rPlotting graphs...",t+1,"/",total,
        sys.stdout.flush()

        g = graph.copy() # makes a copy of the graph so alterations can be made

        deletedFromPlot = [ p for p in range(np) if g.degree(p) == 0 ]

        g.delete_vertices( deletedFromPlot )

        if g.vcount() <= 40:
            g.vs["size"] = [ 30 ] 
#            g.vs["size"] = [ 20 + 4*float(v)/5 for v in g.degree() ] # size varies linearly with node degree, with a constant minimum so isolated nodes won't vanish (10 is iGraph's default size)
#        elif g.vcount() <= 80: 
#            g.vs["size"] = [ 10 + float(v)/15 for v in g.degree() ] 
#22        else: g.vs["size"] = [ 20 for v in g.degree() ] #or else the graphs would be too confusing
        
        g.simplify(combine_edges='sum') # collapses all repeated edges between any two nodes into a single, weighted one

 #       g.es["edge_betweenness"] = g.edge_betweenness() # width of edge connection varies with edge betweenness associated to that edge.
 #       g.es["width"] = [1 + (0.1*e_b) for e_b in g.edge_betweenness()] # modifiers much larger than 0.10 not recommended
        if g.vcount() <= 40:        
            g.es["width"] = [ 2.0 + 2.5*math.sqrt(e_w) for e_w in g.es["weight"] ]  # width of edge takes the square root of edge weight
#        elif g.vcount() <= 80: 
#            g.es["width"] = [ 0.3 + 0.4*math.sqrt(e_w) for e_w in g.es["weight"] ]        
#22       else: g.es["width"] = [ 0.8 + 0.08*math.sqrt(e_w) for e_w in g.es["weight"] ]
 
        g.vs["label"] = [ story.characters[p] for p in range(np) if p not in deletedFromPlot ]


        comm = g.community_multilevel() # partitions nodes into groups/circles based on degree (?)
        
        if g.vcount() <= 10:
            graphLayout = g.layout("circle")
        elif g.vcount() <= 20:
            graphLayout = g.layout("kk")
        else: graphLayout = g.layout("auto")
        
        if not os.path.exists(story.commonFolder):
            os.makedirs(story.commonFolder)
        

        savefile = story.commonPath
                    
        plotTitle = ""
        if story.title != "":
            plotTitle += str(story.title).capitalize()+", "
        if a.allInOne:  
            savefile += story.rangestring + "_aioGraph.png"
            plotTitle += "all-in-one chapt. " + story.rangestring

        else:
            if a.filename:
                savefile += str(a.firstchapter+t)
                plotTitle += "chapter "+str(a.firstchapter+t)

            elif a.singlefile:
                savefile += str(story.chapters[t].match)
                plotTitle += "chapter "+str(story.chapters[t].match)

            savefile += "_Graph.png"

        t += 1


        top = mostImportants(g.vs.degree())
        color_list = ["red","blue","black","SeaGreen","NavyBlue","green","cyan","pink","orange","magenta","magenta","RosyBrown","gold","brown","LightSeaGreen"]
 
        if g.vcount() <= 40:  
            g.vs["color"] = [color_list[0] if vertex.index == top[0]
                         else color_list[1] if vertex.index == top[1]
                         else color_list[2] if vertex.index == top[2]
                         else color_list[3] if vertex.index == top[3]
                         else color_list[4] if vertex.index == top[4]
                         else color_list[5] if vertex.index == top[5]
                         else color_list[6] if vertex.index == top[6]
                         else color_list[7] if vertex.index == top[7]
                         else color_list[8] if vertex.index == top[8]
                         else color_list[9] if vertex.index == top[9]
                         else color_list[10] if vertex.index == top[10]
                         else color_list[11] if vertex.index == top[11]
                         else color_list[12] if vertex.index == top[12]
                         else color_list[13] if vertex.index == top[13]
                         else color_list[14] if vertex.index == top[14]
                         else 'gray'
                 for vertex in g.vs ]
  
  
            g.es["color"] = [g.vs[edge.source]['color'] if (g.vs[edge.source].degree())>(g.vs[edge.target].degree()) else
                     g.vs[edge.target]['color']
                     for edge in g.es]  
  
            plt = plot(g, layout = graphLayout, margin = 70, vertex_label_dist = 1.5, vertex_label_size = 16,
                       background = 'white', target = savefile,
                       #vertex_color= pal_list, 
                       vertex_order_by = ("size", "desc"), edge_order_by = ("width", "desc"),
                       #edge_color = [ pal_list[ indexmin(g.degree, e.tuple[0], e.tuple[1]) ] for e in g.es ] 
                       bbox = [1200,1200])
                       
        else:
            plt = plot(g, target = savefile, bbox = [1200,1200], vertex_label_size = 14, vertex_label_dist = 1.5, vertex_size = 14)
            
                   
 #  edge_color = [ interpolatesColors( [ pal_list[e.tuple[0]] , pal_list[e.tuple[1]] ], 7)[3] for e in g.es] )  # for edge color as interpolation of both vertex colors

    

    
        # Adding a title. Solution by Tamas in https://stackoverflow.com/a/18259578

#       plt.redraw()
        # Grab the surface, construct a drawing context and a TextDrawer
        ctx = cairo.Context(plt.surface)
        ctx.set_font_size(18)
        drawer = TextDrawer(ctx, plotTitle, halign=TextDrawer.CENTER)
        drawer.draw_at(0, 20, width=600)

        # Saves the plot
        plt.save()
 #      plt.show()


    print "\x1b[2K\rPlotting graphs... Done. ",
    sys.stdout.flush()
    print "%d saved in %s"%(t, story.commonFolder)



if a.linePlot or a.heatMap or a.barPlot:

    if a.allInOne:
        nt = 1
    elif a.filename:
        nt = a.lastchapter - a.firstchapter + 1
        chaptersAxis = range(a.lastchapter - a.firstchapter + 1)
        chaptersLabels = [ str(i) for i in chaptersAxis]
    elif a.singlefile:
        nt = len(story.chapters)
        chaptersAxis = range(1,nt+1)
        chaptersLabels = [ chapter.match for chapter in story.chapters]
       
    maxN = np
            
    if (a.top):
        sumArray = critSum(story, a.top)
        if a.top == 'd':
            topCriterion = "Degree"
        elif a.top == 'c':
            topCriterion = "Closeness"
        elif a.top == 'b':
            topCriterion = "Betweenness"
        elif a.top == 'mnc':
            topCriterion = 'Marginal Node Centrality'
        indexes = cull(sumArray, watershedPercentage, maxN)
    else:
        sumArray = critSum(story, 'd') # default ranking by sum of degree centralities, to determine the order of character names in plotting
        indexes = argsortDescending(sumArray)

        
    if a.featuredCharactersFile or a.featuredCharactersList:
        indexes = story.plottedIdx
        plotNames = story.plottedCharacters
        nplotted = len(indexes)
        nfeatured = nplotted

    else:
        plotNames = [ story.characters[idx] for idx in indexes ]
        nplotted = len(indexes)
        nfeatured = 6


    if nplotted == 0:
        print "\n\nError: no characters in plot. Graph plotting unsuccessful.\n\n"
        a.linePlot = a.heatMap = a.barPlot = []
        

    
    if 'ec' in a.linePlot:
        if nt < 2:
            print "Plotting a centrality line graph requires a range of two or more chapters."
        else:                
            linePlotArray = numpy.zeros((len(indexes), nt ))
            plotIdx = 0
            for storyIdx in indexes:
                for t in range(nt):
                    linePlotArray[plotIdx, t] = story.evcentlists[t][storyIdx]
                plotIdx += 1
                
            filename = story.commonPath + story.rangestring + "_EVCentr_Lines"

            plotTitle = "Eigenvector Centrality"
            if a.top:
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"'"
            drawsLineplot(plotTitle, linePlotArray, chaptersAxis, chaptersLabels, '', 'Chapters', plotNames, filename, nfeatured)
            print "Line plot saved in "+filename
    
    if 'jc' in a.linePlot:
        if nt < 2:
            print "Plotting a centrality line graph requires a range of two or more chapters."
        else:                
            linePlotArray = numpy.zeros((len(indexes), nt ))
            plotIdx = 0
            for storyIdx in indexes:
                for t in range(nt):
                    linePlotArray[plotIdx, t] = story.jointEigenvector[np*t + storyIdx]
                plotIdx += 1

            filename = story.commonPath + story.rangestring + "_jointCentr_Lines"
            
            plotTitle = "Joint Centrality"
            if a.top:
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"' (eps="+str(story.epsilon)+")"
            drawsLineplot(plotTitle, linePlotArray, chaptersAxis, chaptersLabels, '', 'Chapters', plotNames, filename, nfeatured)
            print "Line plot saved in "+filename

    
    if 'cc' in a.linePlot:
        if nt < 2:
            print "Plotting a centrality line graph requires a range of two or more chapters."
        else:
            
            linePlotArray = numpy.zeros((nplotted, nt ))
            for x in range(nt):
                plotIdx = 0
                for storyIdx in indexes:
                    linePlotArray[plotIdx, x] = story.conditionalEigenvector[np * x + storyIdx]
                    plotIdx +=1

            filename = story.commonPath + story.rangestring + "_conditionalCentr_Lines"

            plotTitle = "Conditional Centrality"
            if a.top :
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"' (eps="+str(story.epsilon)+")"
                
            drawsLineplot(plotTitle, linePlotArray, chaptersAxis, chaptersLabels, '', 'Chapters', plotNames, filename, nfeatured)
            print "Line plot saved in "+filename

    if 'd' in a.linePlot:
        if nt < 2:
            print "Plotting a centrality line graph requires a range of two or more chapters."
        else:
            
            linePlotArray = numpy.zeros((nplotted, nt ))
            for x in range(nt):
                plotIdx = 0
                for storyIdx in indexes:
                    linePlotArray[plotIdx, x] = story.graphs[x].degree(storyIdx)
                    plotIdx +=1

            filename = story.commonPath + story.rangestring +  "_degrees_Lines"

            plotTitle = "Degree Centrality"
            if a.top :
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"'"
                
            drawsLineplot(plotTitle, linePlotArray, chaptersAxis, chaptersLabels, '', 'Chapters', plotNames, filename, nfeatured)
            print "Line plot saved in "+filename


    if 'c' in a.linePlot:
        if nt < 2:
            print "Plotting a centrality line graph requires a range of two or more chapters."
        else:
            
            linePlotArray = numpy.zeros((nplotted, nt ))
            for x in range(nt):
                plotIdx = 0
                for storyIdx in indexes:
                    linePlotArray[plotIdx, x] = story.graphs[x].closeness(storyIdx)
                    plotIdx +=1

            filename = story.commonPath + story.rangestring +  "_closeness_Lines"

            plotTitle = "Closeness Centrality"
            if a.top :
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"'"
                
            drawsLineplot(plotTitle, linePlotArray, chaptersAxis, chaptersLabels, '', 'Chapters', plotNames, filename, nfeatured)
            print "Line plot saved in "+filename


    if 'b' in a.linePlot:
        if nt < 2:
            print "Plotting a centrality line graph requires a range of two or more chapters."
        else:
            
            linePlotArray = numpy.zeros((nplotted, nt ))
            for x in range(nt):
                plotIdx = 0
                for storyIdx in indexes:
                    linePlotArray[plotIdx, x] = story.graphs[x].betweenness(storyIdx)
                    plotIdx +=1

            filename = story.commonPath + story.rangestring +  "_betw_Lines"

            plotTitle = "Betweenness Centrality"
            if a.top :
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"'"
                
            drawsLineplot(plotTitle, linePlotArray, chaptersAxis, chaptersLabels, '', 'Chapters', plotNames, filename, nfeatured)
            print "Line plot saved in "+filename


    if 'ec' in a.heatMap:
        if nt < 1:
            print "Plotting a centrality heat map requires at least one chapter."
        else:
            heatPlotArray = numpy.zeros((len(indexes), nt ))
            plotIdx = len(indexes)-1
            for storyIdx in indexes:
                for t in range(nt):
                    heatPlotArray[plotIdx, t] = story.evcentlists[t][storyIdx]
                plotIdx -= 1

            filename = story.commonPath + story.rangestring
            
            plotTitle = "Eigenvector Centrality"

            if a.allInOne:
                plotTitle += " (all-in-one)"
                filename += "_EVCentrAIO_Heat"
            else:
                plotTitle += " (chapter-by-chapter)"
                filename += "_EVCentr_Heat"
                
            if a.top :
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"'"

            plotsHeatMap(plotTitle, heatPlotArray, plotNames[::-1], nt, heatPlotArray.min(), heatPlotArray.max(), filename)
            print "Heatmap saved in "+filename


    if 'jc' in a.heatMap:
        if nt < 1:
            print "Plotting a centrality heat map requires at least one chapter."
        else:
            heatPlotArray = numpy.zeros((len(indexes), nt ))
            plotIdx = len(indexes)-1
            for storyIdx in indexes:
                for t in range(nt):
                    heatPlotArray[plotIdx, t] = story.jointEigenvector[np*t + storyIdx]
                plotIdx -= 1

            filename = story.commonPath + story.rangestring
                
            plotTitle = "Joint Centrality"

            if a.allInOne:
                plotTitle += " (all-in-one)"
                filename += "_jointCentrAIO_Heat"
            else:
                plotTitle += " (chapter-by-chapter)"
                filename += "_jointCentr_Heat"
                
            if a.top :
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"' (eps="+str(story.epsilon)+")"

            plotsHeatMap(plotTitle, heatPlotArray, plotNames[::-1], nt, heatPlotArray.min(), heatPlotArray.max(), filename)
            print "Heatmap saved in "+filename


    if 'cc' in a.heatMap:
        if nt < 1:
            print "Plotting a centrality heat map requires at least one chapter."
        else:
            heatPlotArray = numpy.zeros(( nplotted, nt ))
            plotIdx = len(indexes)-1
            for storyIdx in indexes:
                for t in range(nt):
                    heatPlotArray[plotIdx, t] = story.conditionalEigenvector[np*t + storyIdx]
                plotIdx -= 1
                
            filename = story.commonPath + story.rangestring
                
            plotTitle = "Conditional Centrality"

            if a.allInOne:
                plotTitle += " (all-in-one)"
                filename += "_conditionalCentrAIO_Heat"
            else:
                plotTitle += " (chapter-by-chapter)"
                filename += "_conditionalCentr_Heat"

            if a.top:
                filename += "Top.png"
                plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
            else:
                filename += "All.png"
            if story.title != "":
                plotTitle += ",\n" +"'"+str(story.title)+"' (eps="+str(story.epsilon)+")"

            plotsHeatMap(plotTitle, heatPlotArray, plotNames[::-1], nt, heatPlotArray.min(), heatPlotArray.max(), filename)
            print "Heatmap saved in "+filename


    if 'd' in a.heatMap:
        ##degree
        heatPlotArray = numpy.zeros(( nplotted, nt))
        plotIdx = len(indexes)-1
        for storyIdx in indexes:
            t=0
            for g in story.graphs:
                heatPlotArray[plotIdx, t] = g.degree(storyIdx)
                t += 1
            plotIdx -= 1

        filename = story.commonPath + story.rangestring

        plotTitle = "Character Degrees"
        if a.allInOne:
            plotTitle += " (all-in-one)"
            filename += "_degreesAIO_Heat"
        else:
            plotTitle += " (chapter-by-chapter)"
            filename += "_degrees_Heat"

        if a.top:
            filename += "Top.png"
            plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
        else:
            filename += "All.png"
        if story.title != "":
            plotTitle += ",\n" +"'"+str(story.title)+"'"

        plotsHeatMap(plotTitle, heatPlotArray, plotNames[::-1], len(story.graphs), heatPlotArray.min(), heatPlotArray.max(), filename)
        print "Heatmap saved in "+filename


    if 'b' in a.heatMap:
        ##betweenness
        heatPlotArray = numpy.zeros(( nplotted, nt))
        plotIdx = len(indexes)-1
        for storyIdx in indexes:
            t = 0
            for g in story.graphs:
                heatPlotArray[plotIdx, t] = g.betweenness(storyIdx)
                t += 1
            plotIdx -= 1

        filename = story.commonPath + story.rangestring

        plotTitle = "Character Betweenness"
        if a.allInOne:
            plotTitle += " (all-in-one)"
            filename += "_betwAIO_Heat"
        else:
            plotTitle += " (chapter-by-chapter)"
            filename += "_betw_Heat"

        if a.top:
            filename += "Top.png"
            plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
        else:
            filename += "All.png"
        if story.title != "":
            plotTitle += ",\n" +"'"+str(story.title)+"'"

        plotsHeatMap(plotTitle, heatPlotArray, plotNames[::-1], len(story.graphs), heatPlotArray.min(), heatPlotArray.max(), filename)
        print "Heatmap saved in "+filename

    if 'c' in a.heatMap:
        ##closeness
        heatPlotArray = numpy.zeros(( nplotted, nt))
        plotIdx = len(indexes)-1
        for storyIdx in indexes:
            t = 0
            for g in story.graphs:
                heatPlotArray[plotIdx, t] = g.closeness(storyIdx)
                t += 1
            plotIdx -= 1

        filename = story.commonPath + story.rangestring

                
        plotTitle = "Character Closeness"
        if a.allInOne:
            plotTitle += " (all-in-one)"
            filename += "_closeAIO_Heat"
        else:
            plotTitle += " (chapter-by-chapter)"
            filename += "_close_Heat"

        if a.top:
            filename += "Top.png"
            plotTitle += "\nfor top "+str(nplotted)+" characters by "+topCriterion
        else:
            filename += "All.png"
        if story.title != "":
            plotTitle += ",\n" +"'"+str(story.title)+"'"

        plotsHeatMap(plotTitle, heatPlotArray, plotNames[::-1], len(story.graphs), heatPlotArray.min(), heatPlotArray.max(), filename)
        print "Heatmap saved in "+filename       
    
    if 'v' in a.linePlot:
        if np < 2:
            print "Plotting a vitality line graph requires two or more characters."
        else:
            distinguished = argsortDescending(story.absvitalities)[:5]
            linePlotArray = numpy.zeros(np)
            for idx in range(np):
                linePlotArray[idx] = story.freemanIndexWhenEachAbsent[idx]

            savefile = story.commonPath + story.rangestring

            if a.allInOne:
                plotTitle = "All-in-one"
                savefile += "_vitalitiesAIO_Lines"
            else:
                plotTitle = "Chapter-by-chapter"
                savefile += "_vitalities_Lines"
            
            if story.title == "":
                plotTitle += " Vitalities"
            else:
                plotTitle += " Vitalities,\n" + "'"+ str(story.title)+"' (eps="+str(story.epsilon)+")"
                
            plotsVitalitiesLineGraph(plotTitle, linePlotArray, story.characters, story.freemanIndex, distinguished, savefile)


    if 'b' in a.barPlot:

        t = 0
        total = len(story.graphs)
        
        for graph in story.graphs:

            print "\x1b[2K\rPlotting bargraphs...",t+1,"/",total,
            sys.stdout.flush()
            
            dataArray = numpy.array(graph.betweenness())
            namesArray = numpy.array(story.characters)
            
            b_indexes = argsortAscending(dataArray)

            dataArray = dataArray[b_indexes]
            dataArray = dataArray[dataArray > 0]
            
            namesArray = namesArray[b_indexes]
            namesArray = namesArray[-len(dataArray):]


            savefile = story.commonPath
                    
            plotTitle = "Betweenness, "
            if story.title != "":
                plotTitle += str(story.title)+", "
            if a.allInOne:
                savefile += story.rangestring + "_aioBetw_Bars.png"
                plotTitle += "all-in-one chapt. " + story.rangestring

            else:
                if a.filename:
                    savefile += str(a.firstchapter+t)
                    plotTitle += "chapter "+str(a.firstchapter+t)

                elif a.singlefile:
                    savefile += str(story.chapters[t].match)
                    plotTitle += "chapter "+str(story.chapters[t].match)

                savefile += "_Betw_Bars.png"

            t += 1
            drawsBarplot(plotTitle, dataArray, namesArray, savefile)


        print "\x1b[2K\rPlotting bargraphs... Done. ",
        sys.stdout.flush()
        print "%d saved in %s"%(t, story.commonFolder)

    if 'c' in a.barPlot:

        t = 0
        total = len(story.graphs)
        
        for graph in story.graphs:

            print "\x1b[2K\rPlotting bargraphs...",t+1,"/",total,
            sys.stdout.flush()
            
            dataArray = numpy.array(graph.closeness())
            namesArray = numpy.array(story.characters)
            
            b_indexes = argsortAscending(dataArray)

            dataArray = dataArray[b_indexes]
            dataArray = dataArray[dataArray > 0]
            
            namesArray = namesArray[b_indexes]
            namesArray = namesArray[-len(dataArray):]

            savefile = story.commonPath
                
            plotTitle = "Closeness, "
            if story.title != "":
                plotTitle += str(story.title)+", "
            if a.allInOne:
                savefile += story.rangestring + "_aioClose_Bars.png"
                plotTitle += "all-in-one chapt. " + story.rangestring

            else:
                if a.filename:
                    savefile += str(a.firstchapter+t)
                    plotTitle += "chapter "+str(a.firstchapter+t)

                elif a.singlefile:
                    savefile += str(story.chapters[t].match)
                    plotTitle += "chapter "+str(story.chapters[t].match)

                savefile += "_Close_Bars.png"

            t += 1

            drawsBarplot(plotTitle, dataArray, namesArray, savefile)


        print "\x1b[2K\rPlotting bargraphs... Done. ",
        sys.stdout.flush()
        print "%d saved in %s"%(t, story.commonFolder)
            

    if 'd' in a.barPlot:

        t = 0
        total = len(story.graphs)
        
        for graph in story.graphs:
            
            print "\x1b[2K\rPlotting bargraphs...",t+1,"/",total,
            sys.stdout.flush()
            
            dataArray = numpy.array(graph.degree())
            namesArray = numpy.array(story.characters)
            
            b_indexes = argsortAscending(dataArray)

            dataArray = dataArray[b_indexes]
            dataArray = dataArray[dataArray > 0]
            
            namesArray = namesArray[b_indexes]
            namesArray = namesArray[-len(dataArray):]

            savefile = story.commonPath
                
            plotTitle = "Degrees, "
            if story.title != "":
                plotTitle += str(story.title)+", "
            if a.allInOne:
                savefile += story.rangestring + "_aioDegree_Bars.png"
                plotTitle += "all-in-one chapt. " + story.rangestring

            else:
                if a.filename:
                    savefile += str(a.firstchapter+t)
                    plotTitle += "chapter "+str(a.firstchapter+t)

                elif a.singlefile:
                    savefile += str(story.chapters[t].match)
                    plotTitle += "chapter "+str(story.chapters[t].match)

                savefile += "_Degree_Bars.png"

            t += 1

            drawsBarplot(plotTitle, dataArray, namesArray, savefile)

        print "\x1b[2K\rPlotting bargraphs... Done. ",
        sys.stdout.flush()
        print "%d saved in %s"%(t, story.commonFolder)
        
