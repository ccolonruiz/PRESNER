#!/usr/local/bin/python

import argparse
import glob
import os
from collections import defaultdict
from xml.etree import cElementTree


class ClinicalCriteria(object):
    """Criteria in the Track 1 documents."""

    def __init__(self, tid, value):
        """Init."""
        self.tid = tid.strip().upper()
        self.ttype = self.tid
        self.value = value.lower().strip()

    def equals(self, other, mode='strict'):
        """Return whether the current criteria is equal to the one provided."""
        if other.tid == self.tid and other.value == self.value:
            return True
        return False


class ClinicalConcept(object):
    """Named Entity Tag class."""

    def __init__(self, tid, start, end, ttype, text=''):
        """Init."""
        self.tid = str(tid).strip()
        self.start = int(start)
        self.end = int(end)
        self.text = str(text).strip()
        self.ttype = str(ttype).strip()

    def span_matches(self, other, mode='strict'):
        """Return whether the current tag overlaps with the one provided."""
        assert mode in ('strict', 'lenient')
        if mode == 'strict':
            if self.start == other.start and self.end == other.end:
                return True
        else:   
            if (self.end > other.start and self.start < other.end) or \
               (self.start < other.end and other.start < self.end):
                return True
        return False

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        return other.ttype == self.ttype and self.span_matches(other, mode)

    def __str__(self):
        """String representation."""
        return '{}\t{}\t({}:{})'.format(self.ttype, self.text, self.start, self.end)


class RecordTrack2(object):
    """Record for Track 2 class."""

    def __init__(self, file_path):
        """Initialize."""
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()

    @property
    def tags(self):
        return self.annotations['tags']

    @property
    def relations(self):
        return self.annotations['relations']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        with open(self.path) as annotation_file:
            lines = annotation_file.readlines()
            for line_num, line in enumerate(lines):
                if line.strip().startswith('T'):
                    try:
                        tag_id, tag_m, tag_text = line.strip().split('\t')
                    except ValueError:
                        print(self.path, line)
                    if len(tag_m.split(' ')) == 3:
                        tag_type, tag_start, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 4:
                        tag_type, tag_start, _, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 5:
                        tag_type, tag_start, _, _, tag_end = tag_m.split(' ')
                    else:
                        print(self.path)
                        print(line)
                    tag_start, tag_end = int(tag_start), int(tag_end)
                    if tag_type in ['DRUG', 'STRENGTH', 'ROUTE', 'FORM', 'DOSAGE']:
                        annotations['tags'][tag_id] = ClinicalConcept(tag_id,
                                                                      tag_start,
                                                                      tag_end,
                                                                      tag_type,
                                                                      tag_text)

        return annotations

    def _get_text(self):
        """Return the text in the corresponding txt file."""
        path = self.path.replace('.ann', '.txt')
        with open(path) as text_file:
            text = text_file.read()
        return text

    def search_by_id(self, key):
        """Search by id among both tags and relations."""
        try:
            return self.annotations['tags'][key]
        except KeyError():
            try:
                return self.annotations['relations'][key]
            except KeyError():
                return None


class Measures(object):
    """Abstract methods and var to evaluate."""

    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        """Initizialize."""
        assert type(tp) == int
        assert type(tn) == int
        assert type(fp) == int
        assert type(fn) == int
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def precision(self):
        """Compute Precision score."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self):
        """Compute Recall score."""
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1):
        """Compute F1-measure score."""
        assert beta > 0.
        try:
            num = (1 + beta**2) * (self.precision() * self.recall())
            den = beta**2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self):
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)

    def specificity(self):
        """Compute Specificity score."""
        try:
            return self.tn / (self.fp + self.tn)
        except ZeroDivisionError:
            return 0.0

    def sensitivity(self):
        """Compute Sensitivity score."""
        return self.recall()

    def auc(self):
        """Compute AUC score."""
        return (self.sensitivity() + self.specificity()) / 2
    

class SingleCM(object):
    def __init__(self, doc1, doc2, track, mode='strict', verbose=False):
        """Initialize."""
        assert isinstance(doc1, RecordTrack2)
        assert isinstance(doc2, RecordTrack2)
        assert mode in ('strict', 'lenient')
        assert doc1.basename == doc2.basename
        self.scores = {'tags': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
                       'relations': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}}
        self.doc1 = doc1
        self.doc2 = doc2
        self.tags = ['DOSAGE', 'DRUG', 'FORM', 'ROUTE', 'STRENGTH', 'MIS', 'SPU']
        
        gol = [t for t in doc1.tags.values()]
        sys = [t for t in doc2.tags.values()]
        sys_check = [t for t in doc2.tags.values()]
        spu = [t for t in doc2.tags.values()]
        mis = [t for t in doc1.tags.values()]
        
        gol_matched = []
        for s in sys:
            for g in gol:
                if (g.equals(s,mode)):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s in sys_check:
                            sys_check.remove(s)
                            spu.remove(s)
        sys = sys_check
        
        cm = {val: {sub_val: {} for sub_val in self.tags} for val in self.tags[:-2]}
        self.cm = {val: {sub_val: 0 for sub_val in self.tags} for val in self.tags[:-2]}
        self.tp = {val: 0 for val in self.tags[:-2]}
        
        for g in gol:
            for s in sys:
                if (g.span_matches(s,mode)):
                    if s in spu and g.ttype == s.ttype:
                        spu.remove(s)
                    if g in mis and g.ttype == s.ttype:
                        mis.remove(g)
                    
                    if len(cm[g.ttype][s.ttype]):
                        cm[g.ttype][s.ttype].add(s.tid)
                    else:
                        cm[g.ttype][s.ttype] = {s.tid}
                        
        for key in self.cm.keys():
            for sub_key in self.cm[key]:
                self.cm[key][sub_key] = len(cm[key][sub_key])
                if key == sub_key:
                    self.tp[key] = self.cm[key][key]
        
        self.fp = {ttype:len({s.tid for s in sys if s.ttype == ttype})-self.tp[ttype] for ttype in self.tags[:-2]}
        self.fn = {ttype:len({g.tid for g in gol if g.ttype == ttype})-self.tp[ttype] for ttype in self.tags[:-2]}
            
class MultipleCM(object):
    
    def __init__(self, corpora, tag_type=None, mode='strict',
                 verbose=False):
        """Initialize."""
        assert isinstance(corpora, Corpora)
        assert mode in ('strict', 'lenient')
        self.scores = None
        self.tags = ['DOSAGE', 'DRUG', 'FORM', 'ROUTE', 'STRENGTH', 'MIS', 'SPU']
        self.cm = {val: {sub_val: 0 for sub_val in self.tags} for val in self.tags[:-2]}
        
        for g, s in corpora.docs:
                single = SingleCM(g, s, 2, mode, verbose=verbose)
                for k in self.cm.keys():
                    for sub_k in self.cm[k]:
                        self.cm[k][sub_k] += single.cm[k][sub_k]
                    self.cm[k]['MIS'] += single.fn[k]
                    self.cm[k]['SPU'] += single.fp[k]


class SingleEvaluator(object):
    """Evaluate two single files."""

    def __init__(self, doc1, doc2, track, mode='strict', key=None, verbose=False):
        """Initialize."""
        assert isinstance(doc1, RecordTrack2)
        assert isinstance(doc2, RecordTrack2)
        assert mode in ('strict', 'lenient')
        assert doc1.basename == doc2.basename
        self.scores = {'tags': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
                       'relations': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}}
        self.doc1 = doc1
        self.doc2 = doc2
        if key:
            gol = [t for t in doc1.tags.values() if t.ttype == key]
            sys = [t for t in doc2.tags.values() if t.ttype == key]
            sys_check = [t for t in doc2.tags.values() if t.ttype == key]
        else:
            gol = [t for t in doc1.tags.values()]
            sys = [t for t in doc2.tags.values()]
            sys_check = [t for t in doc2.tags.values()]

        gol_matched = []
        for s in sys:
            for g in gol:
                if (g.equals(s,mode)):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s in sys_check:
                            sys_check.remove(s)


        sys = sys_check
        self.scores['tags']['tp'] = len({s.tid for s in sys for g in gol if g.equals(s, mode)})
        self.scores['tags']['fp'] = len({s.tid for s in sys}) - self.scores['tags']['tp']
        self.scores['tags']['fn'] = len({g.tid for g in gol}) - self.scores['tags']['tp']
        self.scores['tags']['tn'] = 0

        if verbose and track == 2:
            tps = {s for s in sys for g in gol if g.equals(s, mode)}
            fps = set(sys) - tps
            fns = set()
            for g in gol:
                if not len([s for s in sys if s.equals(g, mode)]):
                    fns.add(g)
            for e in fps:
                print('FP: ' + str(e))
            for e in fns:
                print('FN:' + str(e))
        if track == 2:
            if key:
                gol = [r for r in doc1.relations.values() if r.rtype == key]
                sys = [r for r in doc2.relations.values() if r.rtype == key]
                sys_check = [r for r in doc2.relations.values() if r.rtype == key]
            else:
                gol = [r for r in doc1.relations.values()]
                sys = [r for r in doc2.relations.values()]
                sys_check = [r for r in doc2.relations.values()]

            gol_matched = []
            for s in sys:
                for g in gol:
                    if (g.equals(s,mode)):
                        if g not in gol_matched:
                            gol_matched.append(g)
                        else:
                            if s in sys_check:
                                sys_check.remove(s)
            sys = sys_check
            self.scores['relations']['tp'] = len({s.rid for s in sys for g in gol if g.equals(s, mode)})
            self.scores['relations']['fp'] = len({s.rid for s in sys}) - self.scores['relations']['tp']
            self.scores['relations']['fn'] = len({g.rid for g in gol}) - self.scores['relations']['tp']
            self.scores['relations']['tn'] = 0
            if verbose:
                tps = {s for s in sys for g in gol if g.equals(s, mode)}
                fps = set(sys) - tps
                fns = set()
                for g in gol:
                    if not len([s for s in sys if s.equals(g, mode)]):
                        fns.add(g)
                for e in fps:
                    print('FP: ' + str(e))
                for e in fns:
                    print('FN:' + str(e))


class MultipleEvaluator(object):
    """Evaluate two sets of files."""

    def __init__(self, corpora, tag_type=None, mode='strict',
                 verbose=False):
        """Initialize."""
        assert isinstance(corpora, Corpora)
        assert mode in ('strict', 'lenient')
        self.scores = None
        self.track2(corpora, tag_type, mode, verbose)

    def track2(self, corpora, tag_type=None, mode='strict', verbose=False):
        """Compute measures for Track 2."""
        self.scores = {'tags': {'tp': 0,
                                'fp': 0,
                                'fn': 0,
                                'tn': 0,
                                'micro': {'precision': 0,
                                          'recall': 0,
                                          'f1': 0},
                                'macro': {'precision': 0,
                                          'recall': 0,
                                          'f1': 0}},
                       'relations': {'tp': 0,
                                     'fp': 0,
                                     'fn': 0,
                                     'tn': 0,
                                     'micro': {'precision': 0,
                                               'recall': 0,
                                               'f1': 0},
                                     'macro': {'precision': 0,
                                               'recall': 0,
                                               'f1': 0}}}
        
        self.tags = ('DRUG', 'STRENGTH', 'ROUTE', 'FORM', 'DOSAGE')
        self.relations = ('Strength-Drug', 'Dosage-Drug', 'Duration-Drug',
                          'Frequency-Drug', 'Form-Drug', 'Route-Drug',
                          'Reason-Drug', 'ADE-Drug')
        for g, s in corpora.docs:
            evaluator = SingleEvaluator(g, s, 2, mode, tag_type, verbose=verbose)
            for target in ('tags', 'relations'):
                for score in ('tp', 'fp', 'fn'):
                    self.scores[target][score] += evaluator.scores[target][score]
                measures = Measures(tp=evaluator.scores[target]['tp'],
                                    fp=evaluator.scores[target]['fp'],
                                    fn=evaluator.scores[target]['fn'],
                                    tn=evaluator.scores[target]['tn'])
                for score in ('precision', 'recall', 'f1'):
                    fn = getattr(measures, score)
                    self.scores[target]['macro'][score] += fn()

        for target in ('tags', 'relations'):
            for key in self.scores[target]['macro'].keys():
                self.scores[target]['macro'][key] = \
                    self.scores[target]['macro'][key] / len(corpora.docs)

            measures = Measures(tp=self.scores[target]['tp'],
                                fp=self.scores[target]['fp'],
                                fn=self.scores[target]['fn'],
                                tn=self.scores[target]['tn'])
            for key in self.scores[target]['micro'].keys():
                fn = getattr(measures, key)
                self.scores[target]['micro'][key] = fn()


def evaluate(corpora, mode='strict', verbose=False):
    """Run the evaluation by considering only files in the two folders."""
    assert mode in ('strict', 'lenient')
    evaluator_s = MultipleEvaluator(corpora, verbose)
    tag_measures_dict = {}
    if corpora.track == 1:
        macro_f1, macro_auc = 0, 0
        
        for tag in evaluator_s.tags:
            macro_f1 += (evaluator_s.scores[(tag, 'met', 'f1')] + evaluator_s.scores[(tag, 'not met', 'f1')])/2
            macro_auc += evaluator_s.scores[(tag, 'met', 'auc')]
        m = Measures(tp=evaluator_s.values['met']['tp'],
                     fp=evaluator_s.values['met']['fp'],
                     fn=evaluator_s.values['met']['fn'],
                     tn=evaluator_s.values['met']['tn'])
        nm = Measures(tp=evaluator_s.values['not met']['tp'],
                      fp=evaluator_s.values['not met']['fp'],
                      fn=evaluator_s.values['not met']['fn'],
                      tn=evaluator_s.values['not met']['tn'])
    else:
        evaluator_l = MultipleEvaluator(corpora, mode='lenient', verbose=verbose)
        for tag in evaluator_s.tags:
            evaluator_tag_s = MultipleEvaluator(corpora, tag, verbose=verbose)
            evaluator_tag_l = MultipleEvaluator(corpora, tag, mode='lenient', verbose=verbose)
            tag_measures_dict[tag] = {}
            tag_measures_dict[tag]['strict'] = evaluator_tag_s.scores['tags']['micro']
            tag_measures_dict[tag]['lenient'] = evaluator_tag_l.scores['tags']['micro']
        tag_measures_dict['Overall'] = {}
        tag_measures_dict['Overall']['strict'] = evaluator_s.scores['tags']
        tag_measures_dict['Overall']['lenient'] = evaluator_l.scores['tags']

        for rel in evaluator_s.relations:
            evaluator_tag_s = MultipleEvaluator(corpora, rel, mode='strict', verbose=verbose)
            evaluator_tag_l = MultipleEvaluator(corpora, rel, mode='lenient', verbose=verbose)
            
        return tag_measures_dict 


class Corpora(object):

    def __init__(self, folder1, folder2, track_num):
        extensions = {1: '*.xml', 2: '*.ann'}
        file_ext = extensions[track_num]
        self.track = track_num
        self.folder1 = folder1
        self.folder2 = folder2
        files1 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder1, file_ext))])
        files2 = set([os.path.basename(f) for f in glob.glob(
            os.path.join(folder2, file_ext))])
        common_files = files1 & files2     
        if not common_files:
            print('ERROR: None of the files match.')

        self.docs = []
        for file in common_files:
            g = RecordTrack2(os.path.join(self.folder1, file))
            s = RecordTrack2(os.path.join(self.folder2, file))
            self.docs.append((g, s))
            
def get_cm(f1, f2, track, mode, verbose):
    """Where the magic begins. 2"""
    corpora = Corpora(f1, f2, track)
    if corpora.docs:
        return MultipleCM(corpora, tag_type=None, mode=mode, verbose=False)


def main(f1, f2, track, verbose):
    """Where the magic begins."""
    corpora = Corpora(f1, f2, track)
    if corpora.docs:
        return evaluate(corpora, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='n2c2: Evaluation script for Track 2')
    parser.add_argument('folder1', help='First data folder path (gold)')
    parser.add_argument('folder2', help='Second data folder path (system)')
    args = parser.parse_args()
    main(os.path.abspath(args.folder1), os.path.abspath(args.folder2), 2, False)
