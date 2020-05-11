import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt    
from itertools import permutations
from data_loader import get_label_coding,load_shj_abstract,load_shj_images
from scipy.stats import sem
from os import mkdir, path
import pandas as pd
import itertools
import argparse

#
# PyTorch implementation of
#  	ALCOVE: An exemplar-based connectionist model of category learning (Kruschke, 1992)

#   Main script runs ALCOVE on stimuli of Shephard, Hovland, and Jenkins (1961)

#  There are a few modifications to Kruschke's original ALCOVE:
#    - with two classes, there is only one binary output
#    - rather the using Kruschke's loss, there are options to maximize
#    the log-likelihood directly (ll loss) or a version of the humble teacher (hinge loss)

class ALCOVE(nn.Module):

	def __init__(self, exemplars, c, phi):
		# Input
		#   exemplars: [ne x dim] rows are exemplars provided to model
		super(ALCOVE, self).__init__()
		self.ne = exemplars.size(0) # number of exemplars
		self.dim = exemplars.size(1) # stimulus dimension
		self.exemplars = exemplars # ne x dim

		# set attention weights to be uniform
		self.attn = torch.nn.Parameter(torch.ones((self.dim,1))/float(self.dim))

		# set association weights to zero
		self.w = torch.nn.Linear(self.ne,1,bias=False)
		self.w.weight = torch.nn.Parameter(torch.zeros((1,self.ne)))

		self.c = c # sharpness parameter (Kruschke uses 6.5 in SHJ simulations)
		self.phi = phi # temperature when making decisions; not included in loss (Kruschke uses 2.0)

	def forward(self,x):
		# Input
		#  x: [dim tensor] a single stimulus
		#
		# Output
		#  output : [tensor scalar] unnormalized log-score (before sigmoid)
		#  prob : [tensor scalar] sigmoid output
		x = x.view(-1,1) # dim x 1
		x_expand = x.expand((-1,self.ne)) # dim x ne
		x_expand = torch.t(x_expand) # ne x dim
		attn_expand = self.attn.expand((-1,self.ne)) # dim x ne
		attn_expand = torch.t(attn_expand) # ne x dim

		# memory/hidden layer is computes the similarity of stimulus x to each exemplar
		hidden = attn_expand * torch.abs(self.exemplars-x_expand) # ne x dim
		hidden = torch.sum(hidden,dim=1) # ne
		hidden = torch.exp(-self.c * hidden) # ne
		hidden = hidden.view((1,-1)) # 1 x ne
		
		# compute the output response
		output = self.w(hidden).view(-1) # tensor scalar
		prob = torch.sigmoid(self.phi*output) # tensor scalar
		return output,prob

class MLP(nn.Module):

	def __init__(self, phi, nhid=8):
		# Input
		#   exemplars: [ne x dim] rows are exemplars provided to model
		super(MLP, self).__init__()
		self.ne = exemplars.size(0) # number of exemplars
		self.dim = exemplars.size(1) # stimulus dimension
		self.nhid = nhid
		self.hid = torch.nn.Linear(self.dim,self.nhid)
		self.out = torch.nn.Linear(self.nhid,1)
		self.phi = phi

	def forward(self,x):
		# Input
		#  x: [dim tensor] a single stimulus
		#
		# Output
		#  output : [tensor scalar] unnormalized log-score (before sigmoid)
		#  prob : [tensor scalar] sigmoid output
		x = x.view(1,-1) # dim x 1
		x = self.hid(x)
		x = torch.tanh(x)
		output = self.out(x)
		prob = torch.sigmoid(self.phi*output) # tensor scalar
		return output,prob

def update_batch(net,exemplars,targets,loss,optimizer):
	# Update the weights using batch SGD for the entire set of exemplars
	#
	# Input
	#   exemplars: [ne x dim tensor] all stimuli/exempalrs in experiment 
	#   targets:   [ne tensor] classification targets (1/0 or 1/-1, depending on loss)
	#   loss: function handle
	#   optimizer : SGD optimizer
	net.zero_grad()
	net.train()
	n_exemplars = exemplars.size(0)
	out = torch.zeros(n_exemplars)
	for j in range(n_exemplars):
		out[j],_ = net.forward(exemplars[j])
	myloss = loss(out, targets)
	myloss.backward()
	optimizer.step()
	if model_type == 'alcove':
		net.attn.data = torch.clamp(net.attn.data, min=0.) # ensure attention is non-negative
	return myloss.cpu().item()

def update_single(net, exemplars, targets, loss, optimizer):
	#net.zero_grad()
	#net.train()
	
	n_exemplars = exemplars.size(0)
	out = torch.zeros(n_exemplars)
	current_loss = torch.zeros(n_exemplars)
	for j in range(n_exemplars):
		net.zero_grad()
		net.train()
	
		out[j],_ = net.forward(exemplars[j])
		myloss = loss(out[j], targets[j])
		myloss.backward(retain_graph=True)
		optimizer.step()
		current_loss[j] = myloss.cpu().item()
		if model_type == 'alcove':
			net.attn.data = torch.clamp(net.attn.data, min=0.) # ensure attention is non-negative
	return torch.sum(current_loss).item()
	

def evaluate(net,exemplars,targets):
	# Compute probability of getting each answer/exemplar right using sigmoid
	# 
	# Input
	#   exemplars: [ne x dim tensor] all stimuli/exempalrs in experiment 
	#   targets:   [ne tensor] classification targets (1/0 or 1/-1, depending on loss)
	#
	# Output
	#   mean probability of correct response
	#   mean accuracy when picking most likely response
	net.eval()
	n_exemplars = exemplars.size(0)
	v_acc = np.zeros(n_exemplars)
	v_prob = np.zeros(n_exemplars)
	for j in range(n_exemplars):
		out,prob = net.forward(exemplars[j])
		out = out.item() # logit
		prob = prob.item() # prob of decision 
		if targets[j].item()==POSITIVE:		
			v_prob[j] = prob
			v_acc[j] = out >= 0	
		elif targets[j].item()==NEGATIVE:
			v_prob[j] = 1-prob
			v_acc[j] = out < 0
	return np.mean(v_prob), 100.*np.mean(v_acc)

def HingeLoss(output, target):
	# Reinterpretation of Kruschke's humble teacher
	#  loss = max(0,1-output * target)
	#
	# Input
	#  output : 1D tensor (raw prediction signal)
	#  target : 1D tensor (must be -1. and 1. labels)
    hinge_loss = 1.-torch.mul(output, target)
    hinge_loss[hinge_loss < 0] = 0.
    return torch.sum(hinge_loss)

def HumbleTeacherLoss(output, target):
	humble_loss = torch.mul(output, target)
	humble_loss[humble_loss > 1] = 1
	humble_loss = (1.-humble_loss)**2
	return .5 * torch.sum(humble_loss)


def train(exemplars,labels,num_epochs,loss_type,typenum,c,phi,track_inc=5,verbose_params=False):
	# Train model on a SHJ problem
	# 
	# Input
	#   exemplars : [n_exemplars x dim tensor] rows are exemplars
	#   labels : [n_exemplars tensor] category labels	
	#   num_epochs : number of passes through exemplar set
	#   loss_type : either 'll' or 'hinge'
	#	track_inc : track loss/output at these intervals
	#   verbose_params : print parameters when you are done
	#
	# Output
	#    trackers for epoch index, probability of right response, accuracy, and loss
	#    each is a list with the same length
	n_exemplars = exemplars.size(0)
	
	if model_type == 'mlp':
		net = MLP(phi)
	elif model_type == 'alcove':
		net = ALCOVE(exemplars,c,phi)
		#df.at[type_tracker:type_tracker+num_rows,'c'] = net.c
	else:
		assert False
		
	#df.at[type_tracker:type_tracker+num_rows,'phi'] = net.phi

	if loss_type == 'll':
		loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
	elif loss_type == 'hinge':
		loss = HingeLoss
	elif loss_type == 'mse':
		loss = torch.nn.MSELoss(reduction='sum')
	elif loss_type == 'humble':
		loss = HumbleTeacherLoss
	else:
		assert False # undefined loss
	
	optimizer = optim.SGD(net.parameters(), lr=lr_association)
	if model_type == 'alcove':
		optimizer = optim.SGD([ {'params': net.w.parameters()}, {'params' : [net.attn], 'lr':lr_attn}], lr=lr_association)

	v_epoch = []
	v_loss = []
	v_acc = []
	v_prob = []
	for epoch in range(1,num_epochs+1):
		#loss_epoch = update_batch(net,exemplars,labels,loss,optimizer)
		loss_epoch = update_single(net,exemplars,labels,loss,optimizer)
		print(loss_epoch)
		if epoch == 1 or epoch % track_inc == 0:
			test_prob,test_acc = evaluate(net,exemplars,labels)
			v_epoch.append(epoch)
			v_loss.append(loss_epoch / float(n_exemplars))
			v_acc.append(test_acc)
			v_prob.append(test_prob)
			print('  epoch ' + str(epoch) + "; train loss " + str(round(v_loss[-1],4)))
			df.at[epoch//track_inc+type_tracker, 'Train Loss'] = round(v_loss[-1],4)
			df.at[epoch//track_inc+type_tracker, 'Train Accuracy'] = round(v_acc[-1],4)
			df.at[epoch//track_inc+type_tracker, 'Probability Correct'] = round(v_prob[-1],4)
			df.at[epoch//track_inc+type_tracker, 'Epoch'] = epoch

	if model_type == 'alcove' and verbose_params:
		print("Attention weights:")
		print(np.transpose(np.round(net.attn.data.numpy(),3)))
		print("Category associations:")
		print(np.round(net.w.weight.data.numpy(),3))

	return v_epoch,v_prob,v_acc,v_loss

def initialize_df(track_inc,num_rows,args):
	# Initializes a DataFrame for translation to .csv
	
	num_epochs = args[4]
	epoch_range = list(range(track_inc,num_epochs+1,track_inc))
	epoch_range.insert(0,1)
	df = pd.DataFrame(index=range(1,(num_rows)*ntype*len(list_exemplars)+1), 
				      columns=['Model','Net','Loss Type','Image Set','LR-Attention',
				   'LR-Association','c','phi','Permutation','Type','Epoch','Train Loss',
				   'Train Accuracy','Probability Correct'])
	df.at[:,'Model'] = args[0]
	df.at[:,'Image Set'] = args[1]
	df.at[:,'Net'] = args[2]
	df.at[:,'Loss Type'] = args[3]
	df.at[:,'LR-Association'] = args[5]
	df.at[:,'LR-Attention'] = args[6]
	df.at[:,'c'] = args[7]
	df.at[:,'phi'] = args[8]
	
	return df

def create_dir(model_type,image_set,net_type,loss_type,_num_epochs,plot):
	# Creates directory and file names for plot/csv storage
	if(image_set == 'abstract'):
		d = 'ab'
		sub_subdir_name = ''
	else:
		d = 'im'
		sub_subdir_name = f'/{image_set}'
		
	if(plot):
		dir_name = 'plots'
		subdir_name = f'{dir_name}/{model_type}_{d}'
		sub_subdir_name = subdir_name + sub_subdir_name
		file_dir = f'{sub_subdir_name}/{net_type}{loss_type}_{num_epochs}' 
		title = f'{model_type} model: {d} stimulus_{net_type}{loss_type}'
	else:
		dir_name = 'csv'
		subdir_name = f'{dir_name}/{model_type}_{d}'
		file_dir = f'{subdir_name}/{net_type}{loss_type}_{num_epochs}' 
		title = None
		
	try:
		mkdir(dir_name)
	except FileExistsError:
		pass
	try:
		mkdir(subdir_name)
	except FileExistsError:
		pass
	try:
		mkdir(sub_subdir_name)
	except FileExistsError:
		pass
		
	return file_dir,title

def create_plot(list_trackers,ntype,title,file_dir):
	A = np.array(list_trackers) # nperms x ntype x 4 tracker types x n_iters
	M = np.mean(A,axis=0) # ntype x 4 tracker types x n_iters
	SE = sem(A,axis=0) # ntype x 4 tracker types x n_iters		
				
	plt.figure(1)
	for i in range(ntype): 
		if viz_se:
			plt.errorbar(M[i,0,:],M[i,1,:],yerr=SE[i,1,:],linewidth=4./(i+1))
		else:
			plt.plot(M[i,0,:],M[i,1,:],linewidth=4./(i+1))
								
	plt.suptitle(title)
	plt.xlabel('Block')
	plt.ylabel('Probability correct')
	plt.legend(["Type " + str(s) for s in range(1,7)])
	plt.savefig(file_dir + '1.png')
				
	plt.figure(2)
	for i in range(ntype):
		if viz_se:
			plt.errorbar(M[i,0,:],M[i,3,:],yerr=SE[i,3,:],linewidth=4./(i+1))  # v is [tracker type x n_iters]
		else:
			plt.plot(M[i,0,:],M[i,3,:],linewidth=4./(i+1))  # v is [tracker type x n_iters]
				
	plt.suptitle(title)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(["Type " + str(s) for s in range(1,7)])
	plt.savefig(file_dir + '2.png')
	plt.show()	

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-m","--model" ,help="Model (alcove or mlp)",type=str)
	parser.add_argument("-d","--dataset",help="Dataset (shj_images_set1, abstract, etc)",type=str)
	parser.add_argument("-n","--net",help="Net type (resnet18, resnet152, vgg11",type=str)
	parser.add_argument("-l","--loss",help="Loss (hinge, ll, mse, humble)",type=str)
	parser.add_argument("-e","--epochs",help="List of # Epochs. Comma delimited",type=str)
	parser.add_argument("--lr_assoc",help="Evenly spaced range of lr_assoc values. Enter: start,end,step",
					 type=str)
	parser.add_argument("--lr_assoc_list",help="Explicitly specify values for lr_assoc. Comma delimited",
					 type=str)
	parser.add_argument("--lr_attn",help="Evenly spaced range of lr_attn values. Enter: start,end,step",
					 type=str)
	parser.add_argument("--lr_attn_list",help="Explicitly specify values for lr_attn. Comma delimited",
					 type=str)
	parser.add_argument("--c",help="Evenly spaced range of c values. Enter: start,end,step",
					 type=str)
	parser.add_argument("--c_list",help="Explicitly specify values for c. Comma delimited",
					 type=str)
	parser.add_argument("--phi",help="Evenly spaced range of phi values. Enter: start,end,step",
					 type=str)
	parser.add_argument("--phi_list",help="Explicitly specify values for phi. Comma delimited",
					 type=str)
	parser.add_argument("--all",help="Run all model-net-loss or model-loss configurations",
					 action="store_true")
	
	args = parser.parse_args()
	
	
	if(args.all):
		model = ['alcove','mlp']
		net = ['resnet18','resnet152','vgg11']
		loss = ['humble','hinge','ll','mse']
	else:
		if(args.model is None):
			model = ['alcove']
		else:
			model = [args.model]
		if(args.net is None):
			net = ['resnet18']
		else:
			net = args.net
		if(args.loss is None):
			loss = ['humble']
		else:
			loss = [args.loss]
	if(args.dataset is None):
		dataset = ['abstract']
	else:
		dataset = [args.dataset]
	if(args.epochs is None):
		epochs = [50,100,150]
	else:
		epochs = [int(item) for item in args.epochs.split(',')]
	if(args.lr_assoc is None):
		if(args.lr_assoc_list is None):
			lr_associations = [0.03]
		else:
			lr_associations = [float(item) for item in args.lr_assoc_list.split(',')]
	else:
		vals = [float(item) for item in args.lr_assoc.split(',')]
		lr_associations = np.arange(vals[0],vals[1]+vals[2],vals[2])
	if(args.lr_attn is None):
		if(args.lr_attn_list is None):
			lr_attns = [0.0033]
		else:
			lr_attns = [float(item) for item in args.lr_attn_list.split(',')]
	else:
		vals = [float(item) for item in args.lr_attn.split(',')]
		lr_attns = np.arange(vals[0],vals[1]+vals[2],vals[2])
	if(args.c is None):
		if(args.c_list is None):
			cs = [6.5]
		else:
			cs = [float(item) for item in args.c_list.split(',')]
	else:
		vals = [float(item) for item in args.c.split(',')]
		cs = np.arange(vals[0],vals[1]+vals[2],vals[2])
	if(args.phi is None):
		if(args.phi_list is None):
			phis = [2]
		else:
			phis = [float(item) for item in args.phi_list.split(',')]
	else:
		vals = [float(item) for item in args.phi.split(',')]
		phis = np.arange(vals[0],vals[1]+vals[2],vals[2])

	plot = False # saves plots when true
	start_over = True # overwrites existing csv files (for the first run) when true
	track_inc = 5 # step size for recording epochs
	
	# create directory for extracted features 
	try:
		mkdir('pickle')
	except FileExistsError:
		pass
	
	# create configurations
	configs_im_mlp = itertools.product(model, dataset, net, loss, epochs,
									lr_associations,lr_attns,phis)
	configs_im_alcove = itertools.product(model, dataset, net, loss, epochs,
								    lr_associations,lr_attns,cs,phis)
	configs_ab_mlp = itertools.product(model, dataset, loss, epochs,
									lr_associations,lr_attns,phis)
	configs_ab_alcove = itertools.product(model, dataset, loss, epochs,
								    lr_associations,lr_attns,cs,phis)
			

	if('alcove' in model and dataset[0] != 'abstract'): # data type = images, model = alcove
		for i, (model_type, image_set, net_type, loss_type, num_epochs,
			 lr_association,lr_attn,c,phi) in enumerate(configs_im_alcove):
			print(f'config {i}: {model_type}, {image_set}, {net_type}, {loss_type} loss, {num_epochs} epochs')
			im_dir = 'data/' + image_set # assuming imagesets in a folder titled data
				
			#lr_association = 0.03 # learning rate for association weights
			#lr_attn = 0.0033 # learning rate for attention weights
			ntype = 6 # number of types in SHJ
			viz_se = False # visualize standard error in plot	
			
			# counters for proper translation to .csv file
			global type_tracker
			type_tracker = 1 
			global num_rows 
			num_rows = (num_epochs // track_inc) + 1
			perm_tracker = 1
			num_rows_p = num_rows*ntype
			
			POSITIVE,NEGATIVE = get_label_coding(loss_type)
			list_perms = list(permutations([0,1,2])) # ways of assigning abstract dimensions to visual ones
			list_exemplars = []
			for p in list_perms:
				exemplars,labels_by_type = load_shj_images(loss_type,net_type,im_dir,p)
					# [n_exemplars x dim tensor],list of [n_exemplars tensor]
				list_exemplars.append(exemplars)
			
			# initialize DataFrame for translation to .csv
			df = initialize_df(track_inc,num_rows,[model_type, image_set, net_type, 
				  loss_type, num_epochs,lr_association,lr_attn,c,phi])
			
			dim = list_exemplars[0].size(1)
			print("Data loaded with " + str(dim) + " dimensions.")
			
			
			# Run ALCOVE on each SHJ problem
			list_trackers = []
			for pidx,exemplars in enumerate(list_exemplars): # all permutations of stimulus dimensions
				tracker = []
				print('Permutation ' + str(pidx))
				df.at[perm_tracker:perm_tracker+num_rows_p,'Permutation'] = pidx
				for mytype in range(1,ntype+1): # from type I to type VI
					print('  Training on type ' + str(mytype))
					df.at[type_tracker:type_tracker+num_rows,'Type'] = mytype
					labels = labels_by_type[mytype-1]
					v_epoch,v_prob,v_acc,v_loss = train(exemplars,labels,num_epochs,loss_type,mytype,c,phi,track_inc)
					tracker.append((v_epoch,v_prob,v_acc,v_loss))
					print("")
					type_tracker += num_rows
				list_trackers.append(tracker)
				perm_tracker += num_rows_p
				
			# create directories/filenames for plots/.csv files and title for plots			
			file_dir,title = create_dir(model_type,image_set,net_type+'_',loss_type,
			   num_epochs,plot)
		
			# plot or save to .csv
			if(plot):
				create_plot(list_trackers,ntype,title,file_dir)
			else:
				if(path.isfile(file_dir + '.csv') and not start_over):
					df.to_csv(file_dir + '.csv',mode='a',header=False)
				else:
					df.to_csv(file_dir + '.csv')
					start_over = False
	if('alcove' in model and dataset[0] == 'abstract'):
		for i, (model_type, image_set, loss_type, num_epochs,
			 lr_association,lr_attn,c,phi) in enumerate(configs_ab_alcove):
			print(f'config {i}: {model_type}, {image_set}, {loss_type} loss, {num_epochs} epochs')
			
			#lr_association = 0.03 # learning rate for association weights
			#lr_attn = 0.0033 # learning rate for attention weights
			ntype = 6 # number of types in SHJ
			viz_se = False # visualize standard error in plot	
			
			# counters for proper translation to .csv file
			type_tracker = 1 
			num_rows = (num_epochs // track_inc) + 1
			perm_tracker = 1
			num_rows_p = num_rows*ntype
			
			POSITIVE,NEGATIVE = get_label_coding(loss_type)
			list_perms = list(permutations([0,1,2]))
			list_exemplars = []
			for p in list_perms:
				exemplars,labels_by_type = load_shj_abstract(loss_type,p) 
					# [n_exemplars x dim tensor],list of [n_exemplars tensor]		
				list_exemplars.append(exemplars)

			# initialize DataFrame for translation to .csv
			df = initialize_df(track_inc,num_rows,[model_type, image_set, None, 
				  loss_type, num_epochs,lr_association,lr_attn,c,phi])
	
			dim = list_exemplars[0].size(1)
			print("Data loaded with " + str(dim) + " dimensions.")
			
			# Run ALCOVE on each SHJ problem
			list_trackers = []
			for pidx,exemplars in enumerate(list_exemplars): # all permutations of stimulus dimensions
				tracker = []
				print('Permutation ' + str(pidx))
				df.at[perm_tracker:perm_tracker+num_rows_p,'Permutation'] = pidx
				for mytype in range(1,ntype+1): # from type I to type VI
					print('  Training on type ' + str(mytype))
					df.at[type_tracker:type_tracker+num_rows,'Type'] = mytype
					labels = labels_by_type[mytype-1]
					v_epoch,v_prob,v_acc,v_loss = train(exemplars,labels,num_epochs,loss_type,mytype,c,phi,track_inc)
					tracker.append((v_epoch,v_prob,v_acc,v_loss))
					print("")
					type_tracker += num_rows
				list_trackers.append(tracker)
				perm_tracker += num_rows_p
				
			# create directories/filenames for plots/.csv files and title for plots
			file_dir,title = create_dir(model_type,image_set,'',loss_type,
			   num_epochs,plot)
		
			# plot or save to .csv
			if(plot):
				create_plot(list_trackers,ntype,title,file_dir)
			else:
				if(path.isfile(file_dir + '.csv') and not start_over):
					df.to_csv(file_dir + '.csv',mode='a',header=False)
				else:
					df.to_csv(file_dir + '.csv')
					start_over = False
	if('mlp' in model and dataset[0] != 'abstract'): # data type = images, model = mlp 
		for i, (model_type, image_set, net_type, loss_type, num_epochs,
			 lr_association,lr_attn,phi) in enumerate(configs_im_mlp):
			print(f'config {i}: {model_type}, {image_set}, {net_type}, {loss_type} loss, {num_epochs} epochs')
			im_dir = 'data/' + image_set # assuming imagesets in a folder titled data
				
			#lr_association = 0.03 # learning rate for association weights
			#lr_attn = 0.0033 # learning rate for attention weights
			c = None
			ntype = 6 # number of types in SHJ
			viz_se = False # visualize standard error in plot	
			
			# counters for proper translation to .csv file
			#type_tracker
			type_tracker = 1 
			 #num_rows 
			num_rows = (num_epochs // track_inc) + 1
			perm_tracker = 1
			num_rows_p = num_rows*ntype
			
			POSITIVE,NEGATIVE = get_label_coding(loss_type)
			list_perms = list(permutations([0,1,2])) # ways of assigning abstract dimensions to visual ones
			list_exemplars = []
			for p in list_perms:
				exemplars,labels_by_type = load_shj_images(loss_type,net_type,im_dir,p)
					# [n_exemplars x dim tensor],list of [n_exemplars tensor]
				list_exemplars.append(exemplars)
				
			# initialize DataFrame for translation to .csv
			df = initialize_df(track_inc,num_rows,[model_type, image_set, net_type, 
				  loss_type, num_epochs,lr_association,lr_attn,None,phi])
			
			dim = list_exemplars[0].size(1)
			print("Data loaded with " + str(dim) + " dimensions.")
			
			# Run ALCOVE on each SHJ problem
			list_trackers = []
			for pidx,exemplars in enumerate(list_exemplars): # all permutations of stimulus dimensions
				tracker = []
				print('Permutation ' + str(pidx))
				df.at[perm_tracker:perm_tracker+num_rows_p,'Permutation'] = pidx
				for mytype in range(1,ntype+1): # from type I to type VI
					print('  Training on type ' + str(mytype))
					df.at[type_tracker:type_tracker+num_rows,'Type'] = mytype
					labels = labels_by_type[mytype-1]
					v_epoch,v_prob,v_acc,v_loss = train(exemplars,labels,num_epochs,loss_type,mytype,c,phi,track_inc)
					tracker.append((v_epoch,v_prob,v_acc,v_loss))
					print("")
					type_tracker += num_rows
				list_trackers.append(tracker)
				perm_tracker += num_rows_p
				
			# create directories/filenames for plots/.csv files and title for plots
			file_dir,title = create_dir(model_type,image_set,net_type+'_',loss_type,
			   num_epochs,plot)
		
			# plot or save to .csv
			if(plot):
				create_plot(list_trackers,ntype,title,file_dir)
			else:
				if(path.isfile(file_dir + '.csv') and not start_over):
					df.to_csv(file_dir + '.csv',mode='a',header=False)
				else:
					df.to_csv(file_dir + '.csv')
					start_over = False
	if('mlp' in model and dataset[0] == 'abstract'): 	# data type = abstract, model = mlp
		for i, (model_type, image_set, loss_type, num_epochs,
			 lr_association,lr_attn,phi) in enumerate(configs_ab_mlp):
			print(f'config {i}: {model_type}, {image_set}, {loss_type} loss, {num_epochs} epochs')
			
			#lr_association = 0.03 # learning rate for association weights
			#lr_attn = 0.0033 # learning rate for attention weights
			c = None
			ntype = 6 # number of types in SHJ
			viz_se = False # visualize standard error in plot	
			
			# counters for proper translation to .csv file
			type_tracker = 1 
			num_rows = (num_epochs // track_inc) + 1
			perm_tracker = 1
			num_rows_p = num_rows*ntype
			
			POSITIVE,NEGATIVE = get_label_coding(loss_type)
			list_perms = list_perms = list(permutations([0,1,2]))
			list_exemplars = []
			for p in list_perms:
				exemplars,labels_by_type = load_shj_abstract(loss_type,p) 
					# [n_exemplars x dim tensor],list of [n_exemplars tensor]		
				list_exemplars.append(exemplars)
			
			# initialize DataFrame for translation to .csv
			df = initialize_df(track_inc,num_rows,[model_type, image_set, None, 
				  loss_type, num_epochs,lr_association,lr_attn,None,phi])
	
			dim = list_exemplars[0].size(1)
			print("Data loaded with " + str(dim) + " dimensions.")
			
			# Run ALCOVE on each SHJ problem
			list_trackers = []
			for pidx,exemplars in enumerate(list_exemplars): # all permutations of stimulus dimensions
				tracker = []
				print('Permutation ' + str(pidx))
				df.at[perm_tracker:perm_tracker+num_rows_p,'Permutation'] = pidx
				for mytype in range(1,ntype+1): # from type I to type VI
					print('  Training on type ' + str(mytype))
					df.at[type_tracker:type_tracker+num_rows,'Type'] = mytype
					labels = labels_by_type[mytype-1]
					v_epoch,v_prob,v_acc,v_loss = train(exemplars,labels,num_epochs,loss_type,mytype,c,phi,track_inc)
					tracker.append((v_epoch,v_prob,v_acc,v_loss))
					print("")
					type_tracker += num_rows
				list_trackers.append(tracker)
				perm_tracker += num_rows_p
				
			# create directories/filenames for plots/.csv files and title for plots
			file_dir,title = create_dir(model_type,image_set,'',loss_type,
			   num_epochs,plot)
		
			# plot or save to .csv
			if(plot):
				create_plot(list_trackers,ntype,title,file_dir)
			else:
				if(path.isfile(file_dir + '.csv') and not start_over):
					df.to_csv(file_dir + '.csv',mode='a',header=False)
				else:
					df.to_csv(file_dir + '.csv')
					start_over = False

