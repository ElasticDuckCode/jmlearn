from jmlearn.tree import BinaryNode, view_tree


def main():
    root = BinaryNode(0)
    root.left = BinaryNode(1)
    root.right = BinaryNode((2,3))
    
    dot = root.view_tree()
    dot.render(directory='/tmp')
    return

if __name__ == "__main__":
    main()