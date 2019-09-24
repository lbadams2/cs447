from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxz")
#E = set("e")
E = set("E")
DROP_E = CONS.union(set("u"))
DOUBLE = set("nptr")
I = set("i")

# Implement your solution here
def buildFST():
    #
    # The states (you need to add more)
    # ---------------------------------------
    # 
    f = FST("q0") # q0 is the initial (non-accepting) state
    f.addState("q1") # a non-accepting state
    f.addState("q_ing") # a non-accepting state
    f.addState("q_precede_e")
    f.addState("q_double")
    f.addState("q_i")
    f.addState("q_etoy")
    #f.addState("q_drop_e")
    f.addState("q_EOW", True) # an accepting state (you shouldn't need any additional accepting states)

    #
    # The transitions (you need to add more):
    # ---------------------------------------
    # transduce every element in this set to itself: 
    f.addSetTransition("q0", AZ, "q1")
    # AZ-E =  the set AZ without the elements in the set E
    f.addSetTransition("q1", AZ-E, "q1")

    # get rid of this transition! (it overgenerates):
    #f.addSetTransition("q1", CONS, "q_ing")

    f.addSetTransition("q1", DROP_E, "q_precede_e")
    #f.addSetTransition("q_precede_e", CONS, "q1")
    f.addSetTransition("q1", VOWS, "q_double")
    #f.addSetTransition("q_double", DOUBLE, "q_ing")

    # map the empty string to ing: 
    f.addTransition("q_ing", "", "ing", "q_EOW")
    f.addTransition("q_precede_e", "e", "", "q_ing")
    f.addTransition("q_double", "n", "nn", "q_ing")
    f.addTransition("q_double", "p", "pp", "q_ing")
    f.addTransition("q_double", "t", "tt", "q_ing")
    f.addTransition("q_double", "r", "rr", "q_ing")
    f.addTransition("q1", "i", "", "q_etoy")
    f.addTransition("q_etoy", "e", "y", "q_ing")

    # Return your completed FST
    return f
    

if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)
