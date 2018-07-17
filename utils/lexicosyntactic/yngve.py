import nltk


class Yngve_calculator():
	def is_terminal(self,tree):
		if type(tree[0]) is nltk.tree.ParentedTree:
			return False
		else:
			return True
	
	def count_rsibs(self,tree):
		rsibs = 0
		if tree.parent():
			par = tree.parent()
			numchildren = len(par)
			posit = tree.parent_index()
			rsibs = numchildren - 1 - posit
		return rsibs
	
	def count_depth(self,tree): #recursive
		depth = 0
		if len(tree.treeposition()) < 1:
			return depth
		else:
			depth += self.count_rsibs(tree)
			depth += self.count_depth(tree.parent())
			return depth
	
	def make_depth_list(self,tree,outlist=[]): #recursive
		if self.is_terminal(tree):
			depth = self.count_depth(tree)
			outlist.append(depth)
		else:
			for subtree in tree:
				self.make_depth_list(subtree,outlist)
		return outlist

