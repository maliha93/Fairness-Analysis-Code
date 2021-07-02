import numpy as np
import matplotlib
import os
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import hex2color

import utils
from utils.io import SMLAResultsReader
from datasets import brazil



if __name__ == '__main__':

	# Whether to save figures. If false, figures are displayed only.
	save_figs = True

	# Location to save the figures (will be created if nonexistent)
	figpath = 'figures/science'

	# Figure format
	fmt = 'png'
	if fmt == 'pdf':
		matplotlib.rc('pdf', fonttype=42)

	# Figure DPI for raster formats
	dpi = 200

	# Paths to results files. Figures will be skipped if data cannot be found.
	# Note that the legend is based off of the EO figure, so it will not be 
	# generated if EO data is unavailable.
	eodds_path  = 'results/science_brazil_eodds_0/science_brazil_eodds.h5'
	di_path     = 'results/science_brazil_di_0/science_brazil_di.h5'
	dp_path     = 'results/science_brazil_dp_0/science_brazil_dp.h5'
	pe_path     = 'results/science_brazil_pe_0/science_brazil_pe.h5'
	eo_path     = 'results/science_brazil_eo_0/science_brazil_eo.h5'

	# Epsilon constants used in experiments
	di_e     =  -0.80
	dp_e     =  0.15 
	eodds_e  =  0.35 
	pe_e     =  0.2 
	eo_e     =  0.2

	# Value of delta used in experiments
	delta = 0.05

	# Constants for rendering figures
	n_total = brazil.load(gpa_cutoff=3.0).training_splits()[0].shape[0]

	# Mapping from model names that will appear on legends
	pprint_map = {
		'SC'              : 'SC',
		'QSC'             : 'QSC',
		'FairlearnSVC'    : 'FL',
		'LinSVC'          : 'SVC$_{linear}$',
		'SGD' 	          : 'SGD',
		'SGD(hinge)'      : 'SGD$_{hinge}$',
		'SGD(log)' 	      : 'SGD$_{log}$',
		'SGD(perceptron)' : 'SGD$_{perc}$',
		'SVC' 	          : 'SVC$_{rbf}$',
		'FairConst'       : 'FC'
	}
	base_smla_names = ['SC', 'QSC']
	base_bsln_names = ['SGD', 'LinSVC', 'SVC']

	# Create the figure directory if nonexistent
	if save_figs and not(os.path.isdir(figpath)):
		os.makedirs(figpath)		



	#############
	#  Helpers  #
	#############

	def save(fig, path, *args, **kwargs):
		if not(os.path.isdir(figpath)):
			os.makedirs(figpath)
		path = os.path.join(figpath, path)
		print('Saving figure to \'%s\'' % path)
		fig.savefig(path, *args, **kwargs)

	def get_ls(name):
		if name == 'SC':
			return ':'
		elif name == 'QSC':
			return '--'
		return '-'

	def get_lw(name):
		if name == 'SC':
			return 2
		elif name == 'QSC':
			return 2
		return 1

	def get_samples(results, dpct, e, include_fairlearn=False, include_fairconst=False):
		''' Helper for filtering results files. '''
		_smla_names = base_smla_names
		_bsln_names = base_bsln_names.copy()
		if include_fairlearn:
			_bsln_names.append('FairlearnSVC')
		if include_fairconst:
			_bsln_names.append('FairConst')

		# Get the SMLA samples
		smla_samples = []
		smla_names   = []
		psmla_names  = []
		for nm in _smla_names:
			smla_names.append(nm)
			psmla_names.append(pprint_map[nm])
			sample = results.extract(['accept','loss_test', 'co_0_mean', 'data_pct'], name=nm, data_pct=dpct, e=e)
			smla_samples.append(sample)

		# get the baseline samples (note different versions of SGD)
		bsln_samples = []
		bsln_names   = []
		pbsln_names  = []
		for nm in _bsln_names:
			if nm == 'SGD':
				for loss in ['log','perceptron','hinge']:
					bsln_names.append(nm)
					pbsln_names.append(pprint_map[nm+('(%s)'%loss)])
					sample = results.extract(['accept','loss_test', 'co_0_mean', 'data_pct'], name=nm, data_pct=dpct, loss=loss)
					bsln_samples.append(sample)
			elif nm == 'FairlearnSVC':
				fl_e_vals = np.unique(results._store['method_parameters/FairlearnSVC']['fl_e'])
				for fl_e in fl_e_vals:
					bsln_names.append(nm + ('(%.2f)'%fl_e))
					pbsln_names.append(pprint_map[nm] + ('$_{%.2f}$'%fl_e))
					sample = results.extract(['accept','loss_test', 'co_0_mean', 'data_pct'], name=nm, data_pct=dpct, fl_e=fl_e)
					bsln_samples.append(sample)
			elif nm == 'FairConst':
				cov_vals = np.unique(results._store['method_parameters/FairConst']['cov'])
				for cov in cov_vals:
					bsln_names.append(nm + ('(%.2f)'%cov))
					pbsln_names.append(pprint_map[nm] + ('$_{%.2f}$'%cov))
					sample = results.extract(['accept','loss_test', 'co_0_mean', 'data_pct'], name=nm, data_pct=dpct, cov=cov)
					bsln_samples.append(sample)
			elif nm == 'SVC':
				continue
			else:
				bsln_names.append(nm)
				pbsln_names.append(pprint_map[nm])
				sample = results.extract(['accept','loss_test', 'co_0_mean', 'data_pct'], name=nm, data_pct=dpct)
				bsln_samples.append(sample)
		is_smla = np.array([True]*len(smla_names) + [False]*len(bsln_names))
		return is_smla, (smla_names, psmla_names, smla_samples), (bsln_names, pbsln_names, bsln_samples)


	def get_brazil_stats(path, e, include_fairlearn=False, include_fairconst=False):
		''' Helper for extracting resutls from brazil results files. '''
		results = SMLAResultsReader(path)
		results.open()

		dpcts = np.array(sorted(results.extract(['data_pct']).data_pct.unique()))
		nvals = np.array(sorted(np.floor(dpcts * n_total).astype(int)))
		is_smla, (smla_names, psmla_names, smla_samples), (bsln_names, pbsln_names, bsln_samples) = get_samples(results, dpcts.max(), e, include_fairlearn=include_fairlearn, include_fairconst=include_fairconst)
		all_samples = smla_samples + bsln_samples
		mnames  = np.array(smla_names + bsln_names)
		pmnames = np.array(psmla_names + pbsln_names)
		
		# Compute statistics and close the results file
		arates, arates_se = [], [] # Acceptance rates and SEs
		frates, frates_se = [], [] # Failure rates ans SEs (rate that accepted solutions have g(theta) > 0 on the test set)
		lrates, lrates_se = [], [] # Test set error and SEs
		for _dpct in dpcts:
			_, (_,_,_smla_samples), (_,_,_bsln_samples) = get_samples(results, _dpct, e, include_fairlearn=include_fairlearn, include_fairconst=include_fairconst)
			_arates, _arates_se = [], []
			_frates, _frates_se = [], []
			_lrates, _lrates_se = [], []
			for s in _smla_samples + _bsln_samples:
				accepts  = 1 * s.accept
				_arates.append(np.mean(accepts))
				_arates_se.append(np.std(accepts,ddof=1)/np.sqrt(len(accepts)))

				failures = 1 * np.logical_and(s.co_0_mean>0, s.accept)
				_frates.append(np.mean(failures))
				_frates_se.append(np.std(failures,ddof=1)/np.sqrt(len(failures)))

				if any(s.accept):
					losses = s.loss_test[s.accept]
					_lrates.append(np.mean(losses))
					_lrates_se.append(np.std(losses,ddof=1)/np.sqrt(len(losses)))
				else:
					_lrates.append(np.nan)
					_lrates_se.append(np.nan)
			arates.append(_arates)
			arates_se.append(_arates_se)
			frates.append(_frates)
			frates_se.append(_frates_se)
			lrates.append(_lrates)
			lrates_se.append(_lrates_se)
		arates    = np.array(arates)
		arates_se = np.array(arates_se)
		frates    = np.array(frates)
		frates_se = np.array(frates_se)
		lrates    = np.array(lrates)
		lrates_se = np.array(lrates_se)
		results.close()

		# Assign colors to each method
		# This part is a hack to get reasonable colors for each method. If more methods are
		#   added this section should be changed.
		n_smla, i_smla = len(smla_names), 0
		colors = []
		for smla, nm in zip(is_smla, mnames):
			if smla:
				colors.append(hex2color('#4daf4a'))
			# if smla:
			# 	colors.append(hex2color('#377eb8'))
			elif nm.startswith('FairlearnSVC'):
				colors.append(hex2color('#377eb8'))
				# colors.append(hex2color('#ff7f00'))
			elif nm.startswith('FairConst'):
				colors.append(hex2color('#984ea3'))
			else:
				colors.append(hex2color('#e41a1c'))
		
		out = {
			'mnames'    : mnames,
			'pmnames'   : pmnames,
			'colors'    : colors,
			'nvals'     : nvals,
			'is_smla'   : is_smla,
			'arate_v_n'    : arates,
			'arate_se_v_n' : arates_se,
			'frate_v_n'    : frates,
			'frate_se_v_n' : frates_se,
			'lrate_v_n'    : lrates,
			'lrate_se_v_n' : lrates_se,
			'acc_v_n'      : 100 * (1-lrates),
			'acc_se_v_n'   : 100 * (lrates_se)
		}
		return out



	# ######################################################
	# # Report figure, showing failure rates for DP and DI #
	# ######################################################

	# if not(os.path.exists(dp_path)):
	# 	print('No results found at path \'%s\'. Report figure skipped.' % dp_path)
	# elif not(os.path.exists(di_path)):
	# 	print('No results found at path \'%s\'. Report figure skipped.' % di_path)
	# else:
	# 	rfig, (rax_dp, rax_di) = plt.subplots(1,2, figsize=(11,3.5))
	# 	# Plot Demographic Parity failure rates (report)
	# 	D = get_brazil_stats(dp_path, dp_e, include_fairlearn=True, include_fairconst=True)
	# 	frates    = D['frate_v_n']
	# 	frates_se = D['frate_se_v_n']
	# 	mnames    = D['mnames']
	# 	colors    = D['colors']
	# 	nvals     = D['nvals']
	# 	for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
	# 		if mn.endswith('TTestSC'):
	# 			continue
	# 		rax_dp.plot(nvals, fr, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
	# 		rax_dp.fill_between(nvals, (fr+se), (fr-se), color=c, alpha=0.25, linewidth=0)
	# 	rax_dp.set_title('Demographic Parity')
	# 	rax_dp.axhline(delta, color='k', linestyle=':')
	# 	rax_dp.set_xscale("log")
	# 	rax_dp.set_ylim((-np.max(frates)*0.05, np.max(frates)*1.10))
	# 	rax_dp.set_xlim(right=max(nvals))
	# 	rax_dp.set_xlabel('Number of Training Samples', labelpad=0)
	# 	rax_dp.set_ylabel('Probability of Undesirable Behavior')
	# 	rax_dp.spines['right'].set_visible(False)
	# 	rax_dp.spines['top'].set_visible(False)

	# 	# Plot Disparate Impact failure rates (report)
	# 	D = get_brazil_stats(di_path, di_e, include_fairconst=True, include_fairlearn=True)
	# 	frates    = D['frate_v_n']
	# 	frates_se = D['frate_se_v_n']
	# 	mnames    = D['mnames']
	# 	colors    = D['colors']
	# 	nvals     = D['nvals']
	# 	for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
	# 		if mn.endswith('QNDLC'):
	# 			continue
	# 		rax_di.plot(nvals, fr, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
	# 		rax_di.fill_between(nvals, (fr+se), (fr-se), color=c, alpha=0.25, linewidth=0)
	# 	rax_di.set_title('Disparate Impact')
	# 	rax_di.set_xlabel('Number of Training Samples', labelpad=0)
	# 	rax_di.axhline(delta, color='k', linestyle=':')
	# 	rax_di.set_xscale("log")
	# 	rax_di.set_ylim((-np.max(frates)*0.05, np.max(frates)*1.10))
	# 	rax_di.set_xlim(right=max(nvals))
	# 	rax_di.spines['right'].set_visible(False)
	# 	rax_di.spines['top'].set_visible(False)

	# 	# Display/save the figure
	# 	if save_figs:
	# 		save(rfig,'brazil_di_dp_failure_rates.%s' % fmt, dpi=dpi)
	# 	else:
	# 		fig.show()



	####################################################################
	# DemographicParity: Accuracy, Acceptance Rates, and Failure Rates #
	####################################################################

	if not(os.path.exists(dp_path)):
		print('No results found at path \'%s\'. Skipped.' % dp_path)
	else:
		D = get_brazil_stats(dp_path, dp_e, include_fairlearn=True, include_fairconst=True)
		arates = D['arate_v_n']
		arates_se = D['arate_se_v_n']
		frates = D['frate_v_n']
		frates_se = D['frate_se_v_n']
		acc_v_n = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals = D['nvals']
		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 2))

		# Plot accuracy values
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_acc.set_xlabel('Training Samples', labelpad=-0.5)
		ax_acc.set_ylabel('Accuracy')
		ax_acc.set_ylim((33,67))
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot acceptance rate
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=-0.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot failure rate
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xlabel('Training Samples', labelpad=-0.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-np.max(frates)*5, np.max(frates)*110))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Finalize the figure and display/save
		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.22, wspace=0.3, top=0.96, left=0.055, right=0.98)
		if save_figs:
			save(fig,'arate_bqf_v_n_dp.%s' % fmt, dpi=dpi)
		else:
			fig.show()



	##################################################################
	# DisparateImpact: Accuracy, Acceptance Rates, and Failure Rates #
	##################################################################

	if not(os.path.exists(di_path)):
		print('No results found at path \'%s\'. Skipped.' % di_path)
	else:
		D = get_brazil_stats(di_path, di_e, include_fairconst=True, include_fairlearn=True)
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals  = D['nvals']
		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 2))

		# Plot accuracy values
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_acc.set_xlabel('Training Samples', labelpad=-0.5)
		ax_acc.set_ylabel('Accuracy')
		ax_acc.set_ylim((33,67))
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# legend_data = []
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=-0.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot failure rates
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=-0.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_ylim((-np.max(frates)*5, np.max(frates)*110))
		ax_fr.set_xscale("log")
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Finalize the figure and display/save
		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.22, wspace=0.3, top=0.96, left=0.055, right=0.98)
		if save_figs:
			save(fig,'arate_bqf_v_n_di.%s' % fmt, dpi=dpi)
		else:
			fig.show()



	################################################################
	# EqualizedOdds: Accuracy, Acceptance Rates, and Failure Rates #
	################################################################

	if not(os.path.exists(eodds_path)):
		print('No results found at path \'%s\'. Skipped.' % eodds_path)
	else:
		D = get_brazil_stats(eodds_path, eodds_e, include_fairlearn=True, include_fairconst=True)
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals  = D['nvals']
		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 2))

		# Plot accuracy values
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_acc.set_xlabel('Training Samples', labelpad=-0.5)
		ax_acc.set_ylabel('Accuracy')
		ax_acc.set_ylim((33,67))
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot acceptance rates
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=-0.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot failure rates
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=-0.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-np.max(frates)*5, np.max(frates)*110))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Finalize the figure and display/save
		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.22, wspace=0.3, top=0.96, left=0.055, right=0.98)
		if save_figs:
			save(fig,'arate_bqf_v_n_eodds.%s' % fmt, dpi=dpi)
		else:
			fig.show()



	###################################################################
	# EqualOpportunity: Accuracy, Acceptance Rates, and Failure Rates #
	###################################################################

	if not(os.path.exists(eo_path)):
		print('No results found at path \'%s\'. Skipped.' % eo_path)
	else:
		D = get_brazil_stats(eo_path, eo_e, include_fairlearn=True, include_fairconst=True)
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals  = D['nvals']
		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11, 2))

		# Plot accuracy values
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
		ax_acc.set_xlabel('Training Samples', labelpad=-0.5)
		ax_acc.set_ylabel('Accuracy')
		ax_acc.set_ylim((33,67))
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot acceptance rates
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=-0.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot failure rates
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xlabel('Training Samples', labelpad=-0.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.set_xscale("log")
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.set_ylim((-np.max(frates)*5, np.max(frates)*110))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Finalize the figure and display/save
		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.22, wspace=0.3, top=0.96, left=0.055, right=0.98)
		if save_figs:
			save(fig,'arate_bqf_v_n_eo.%s' % fmt, dpi=dpi)
		else:
			fig.show()



	#####################################################################
	# PredictiveEquality: Accuracy, Acceptance Rates, and Failure Rates #
	#####################################################################

	if not(os.path.exists(pe_path)):
		print('No results found at path \'%s\'. Skipped.' % pe_path)
	else:
		D = get_brazil_stats(pe_path, pe_e, include_fairlearn=True, include_fairconst=True)
		arates     = D['arate_v_n']
		arates_se  = D['arate_se_v_n']
		frates     = D['frate_v_n']
		frates_se  = D['frate_se_v_n']
		acc_v_n    = D['acc_v_n']
		acc_se_v_n = D['acc_se_v_n']
		mnames = D['mnames']
		colors = D['colors']
		nvals  = D['nvals']
		fig, (ax_acc, ax_ar, ax_fr) = plt.subplots(1, 3, figsize=(11,2))

		# Plot accuracy values and populate the legend
		legend_data, added = [], []
		for mn,c,acc,acc_se in zip(mnames[::-1],colors[::-1],(acc_v_n.T)[::-1],(acc_se_v_n.T)[::-1]):
			line = ax_acc.plot(nvals, acc, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_acc.fill_between(nvals, acc+acc_se, acc-acc_se, alpha=0.2, color=c, linewidth=0)
			if mn.endswith('TTestSC') and not('quasi-Seldonian Classification' in added):
				added.append('quasi-Seldonian Classification')
				legend_data.append(line)
			elif mn.endswith('HoeffdingSC') and not('Seldonian Classification' in added):
				added.append('Seldonian Classification')
				legend_data.append(line)
			elif mn.startswith('Fairlearn') and not('Fairlearn' in added):
				added.append('Fairlearn')
				legend_data.append(line)
			elif mn.startswith('FairConst') and not('Fairness Constraints' in added):
				added.append('Fairness Constraints')
				legend_data.append(line)
			elif not(mn.endswith('HoeffdingSC') or mn.startswith('Fairlearn') or mn.startswith('FairConst')) and not('Standard' in added):
				added.append('Standard')
				legend_data.append(line)
		legend_data, added = legend_data[::-1], added[::-1]
		ax_acc.set_xlabel('Training Samples', labelpad=-0.5)
		ax_acc.set_ylabel('Accuracy')
		ax_acc.set_ylim((33,67))
		ax_acc.set_xscale("log")
		ax_acc.set_xlim(right=max(nvals))
		ax_acc.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot acceptance rates
		for mn,c,ar,se in zip(mnames[::-1],colors[::-1],(arates.T)[::-1], (arates_se.T)[::-1]):
			ax_ar.plot(nvals, ar*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_ar.fill_between(nvals, 100*(ar+se), 100*(ar-se), alpha=0.25, linewidth=0, color=c)
		ax_ar.set_xlabel('Training Samples', labelpad=-0.5)
		ax_ar.set_ylabel('Solution Rate', labelpad=-3)
		ax_ar.set_xscale("log")
		ax_ar.set_xlim(right=max(nvals))
		ax_ar.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Plot failure rates
		for mn, c,fr,se in zip(mnames[::-1],colors[::-1],(frates.T)[::-1], (frates_se.T)[::-1]):
			ax_fr.plot(nvals, fr*100, c=c, ls=get_ls(mn), lw=get_lw(mn))[0]
			ax_fr.fill_between(nvals, (fr+se)*100, (fr-se)*100, color=c, linewidth=0, alpha=0.25)
		ax_fr.set_xlabel('Training Samples', labelpad=-0.5)
		ax_fr.set_ylabel('Failure Rate')
		ax_fr.axhline(delta*100, color='k', linestyle=':')
		ax_fr.set_xscale("log")
		ax_fr.set_ylim((-np.max(frates)*5, np.max(frates)*110))
		ax_fr.set_xlim(right=max(nvals))
		ax_fr.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

		# Finalize the figure and display/save
		for ax in [ax_acc, ax_ar, ax_fr]:
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
		fig.subplots_adjust(bottom=0.22, wspace=0.3, top=0.96, left=0.055, right=0.98)
		if save_figs:
			save(fig,'arate_bqf_v_n_pe.%s' % fmt, dpi=dpi)
		else:
			fig.show()

		#####################################
		# Figure containing the legend only #
		#####################################

		fig = plt.figure(figsize=(9.75,0.3))
		fig.legend(legend_data, added, 'center', fancybox=True, ncol=len(legend_data), columnspacing=1, fontsize=11, handletextpad=0.5)
		save(fig, 'legend.%s' % fmt, dpi=dpi)