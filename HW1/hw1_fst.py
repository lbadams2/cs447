from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxz")
#E = set("e")
E = set("e")
DROP_E = CONS.union(set("u"))
DOUBLE = set("nptr")
I = set("i")
Y = set("y")
#CONS_PAIRS = {str(c1 + c2) for c1 in CONS for c2 in CONS}


# Implement your solution here
def buildFST():
    #
    # The states (you need to add more)
    # ---------------------------------------
    # 
    f = FST("q0") # q0 is the initial (non-accepting) state
    f.addState("q1") # a non-accepting state
    f.addState("q2")
    f.addState("q_ing") # a non-accepting state
    f.addState("q_precede_e")
    f.addState("q_double")
    f.addState("q_double_cons")
    f.addState("q_vow_cons")
    f.addState("q_e_cons")
    f.addState("q_double_e")
    f.addState("q_i")
    f.addState("q_e")
    f.addState("q_ea")
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
    f.addSetTransition("q1", E, "q_e")
    f.addSetTransition("q_e", E, "q_ing")
    f.addSetTransition("q_e", E, "q_double_e")
    f.addSetTransition("q_double_e", CONS-set("z"), "q_ing")
    f.addSetTransition("q_double_e", set("z"), "q_precede_e")
    f.addSetTransition("q_e", set("rn"), "q_ing")
    #f.addSetTransition("q_e", set("a"), "q_ea")
    #f.addSetTransition("q_ea", CONS, "q_ing")
    #f.addSetTransition("q_ea", DROP_E, "q_precede_e")
    f.addSetTransition("q_e", AZ-E, "q2")
    f.addSetTransition("q2", AZ, "q2")
    f.addSetTransition("q2", DROP_E, "q_precede_e")
    f.addSetTransition("q2", CONS.union(Y), "q_ing")
    # get rid of this transition! (it overgenerates):
    # maybe try string of consecutive consonants here
    f.addSetTransition("q1", CONS, "q_double_cons")
    f.addSetTransition("q_double_cons", CONS.union(Y), "q_ing")

    f.addSetTransition("q1", DROP_E, "q_precede_e")
    #f.addSetTransition("q_precede_e", CONS, "q1")
    f.addSetTransition("q1", VOWS-E, "q_double")
    
    #f.addSetTransition("q_e_cons", set("rn"), "q_ing")
    #f.addSetTransition("q_e_cons", CONS-set("rn"), "q_double")
    f.addSetTransition("q1", VOWS, "q_vow_cons")
    VOW_CONS = CONS-DOUBLE
    f.addSetTransition("q_vow_cons", VOW_CONS.union(Y), "q_ing")
    #f.addSetTransition("q1", E, "q_double_e")
    #f.addSetTransition("q_double_e", E, "q_ing")
    
    #f.addSetTransition("q_double", DOUBLE, "q_ing")

    # map the empty string to ing: 
    f.addTransition("q_ing", "", "ing", "q_EOW")
    f.addTransition("q_precede_e", "e", "", "q_ing")
    f.addTransition("q_double", "n", "nn", "q_ing")
    f.addTransition("q_double", "p", "pp", "q_ing")
    f.addTransition("q_double", "t", "tt", "q_ing")
    f.addTransition("q_double", "r", "rr", "q_ing")
    f.addTransition("q_e", "t", "tt", "q_ing")
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
