from graphviz import Digraph


class BinaryNode:
    '''Class for creating standard binary tree nodes.
    
    >>> x = BinaryNode(1)
    >>> x.left = BinaryNode(2)
    >>> x.right = BinaryNode(3)
    '''
    
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value
        
    def view_tree(self):
        view_tree(self)
        
        

def view_tree(root):
    
    def add_edges(tree, dot=None):
        if dot is None:
            dot = Digraph()
            dot.node(name=str(tree), label=str(tree.value))
            
        if tree.left:
            dot.node(name=str(tree.left), label=str(tree.left.value))
            dot.edge(str(tree), str(tree.left))
            dot = add_edges(tree.left, dot=dot)
        
        if tree.right:
            dot.node(name=str(tree.right), label=str(tree.right.value))
            dot.edge(str(tree), str(tree.right))
            dot = add_edges(tree.right, dot=dot)
        
        return dot

    dot = add_edges(root)
    return dot
