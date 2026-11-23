"""Validate Core 1.1 primary artifacts without importing or executing Core code."""
from __future__ import annotations

import math
from pathlib import Path
import shutil

from ..common import finite_number, inside_file, read_csv, read_json, validate_predictions


def _component(value):
    return isinstance(value, str) and value not in ('', '.', '..') and not any(c in value for c in '/\\:')


def _accuracy(value):
    return finite_number(value) and 0 <= value <= 1


def _same_score(actual, expected):
    return _accuracy(actual) and _accuracy(expected) and abs(actual - expected) <= 1e-12


def _plain_tree(source, destination):
    """Copy a tree only after checking every descendant, including directory links."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Expected a real artifact directory')
    for path in source.rglob('*'):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Nested artifact links or special files are not accepted')
    shutil.copytree(source, destination)


def _copy_member_artifacts(core, member_dir, node, destination):
    """Materialize only Core's documented link to this node's own model directory."""
    art = member_dir / 'artifacts'
    if not art.exists() and not art.is_symlink():
        return False
    if art.is_symlink():
        expected = core / 'runs' / node / 'out' / 'model'
        for path in (expected, *expected.parents):
            if path == core:
                break
            if path.is_symlink():
                raise ValueError('Model link target has a linked ancestor')
        if art.resolve(strict=True) != expected.resolve(strict=True):
            raise ValueError('Member artifacts must point to its own Core run/model directory')
        art = expected
    _plain_tree(art, destination)
    return any(p.is_file() for p in destination.rglob('*'))


def _validate_best(core, status, sample, target, classes):
    node, score = status.get('best_node_id'), status.get('best_score')
    if not _component(node) or not _accuracy(score):
        raise ValueError('Core 1.1 requires a valid best node and accuracy')
    meta = read_json(inside_file(core, 'best/score.json'))
    if not isinstance(meta, dict) or meta.get('node_id') != node or not _same_score(meta.get('score'), score):
        raise ValueError('Best score.json disagrees with status')
    if meta.get('higher_is_better') is not True:
        raise ValueError('Best score direction must maximize accuracy')
    if not inside_file(core, 'best/solution.py').read_text(encoding='utf-8').strip():
        raise ValueError('Missing best full solution script')
    pred = validate_predictions(inside_file(core, 'best/predictions.csv'), sample, target, classes)
    return node, score, pred


def collect_core11(core, destination, status, partial, sample, target, classes):
    """Return verified selected-result metadata, model presence and delivery mode.

    Only an interrupted/failed run that never wrote final/ may yield best_partial.
    A malformed final result is never silently replaced by a different candidate.
    """
    best_node, best_score, best_pred = _validate_best(core, status, sample, target, classes)
    final = core / 'final'
    if not final.exists() and not final.is_symlink():
        if not partial or status.get('final_available') is not False:
            raise ValueError('Core final result is missing; no successful-run fallback is allowed')
        _plain_tree(core / 'best', destination / 'best')
        shutil.copyfile(core / 'best/predictions.csv', destination / 'predictions.csv')
        models = destination / 'best/artifacts'
        present = models.is_dir() and any(p.is_file() for p in models.rglob('*'))
        return {'source': 'best_partial', 'kind': 'single', 'method': 'single', 'metric': 'accuracy',
                'validation_score': best_score, 'node_ids': [best_node],
                'reason': 'Core stopped before finalization; retained its best validated single result'}, present, 'full_script'

    if status.get('final_available') is not True:
        raise ValueError('A final directory exists but Core did not declare it available')
    source = read_json(inside_file(core, 'final/source.json'))
    if not isinstance(source, dict) or source.get('available') is not True:
        raise ValueError('Final source is not an available result')
    kind = source.get('kind')
    if kind not in ('single', 'ensemble'):
        raise ValueError('Unsupported final result kind')
    for key in ('kind', 'method', 'metric'):
        if 'final_' + key not in status or status['final_' + key] != source.get(key):
            raise ValueError('Final source disagrees with status: ' + key)
    if source.get('metric') not in (None, 'accuracy'):
        raise ValueError('Final selection metric disagrees with course accuracy')
    validation = source.get('validation')
    if validation is not None and not isinstance(validation, dict):
        raise ValueError('Invalid final validation record')
    selected_score = (validation or {}).get('score')
    if selected_score is not None and not _accuracy(selected_score):
        raise ValueError('Invalid final validation accuracy')
    declared = status.get('final_validation_score')
    if 'final_validation_score' not in status or (declared is None) != (selected_score is None):
        raise ValueError('Missing or mismatched final validation score')
    if declared is not None and not _same_score(declared, selected_score):
        raise ValueError('Final validation score disagrees with status')
    if kind == 'ensemble' and (source.get('metric') != 'accuracy' or selected_score is None):
        raise ValueError('An ensemble primary result needs an accuracy validation record')
    if not inside_file(core, 'final/combine.py').read_text(encoding='utf-8').strip():
        raise ValueError('Missing final combination script')
    primary = validate_predictions(inside_file(core, 'final/predictions.csv'), sample, target, classes)
    members = source.get('members')
    if not isinstance(members, list) or not members or (kind == 'single') != (len(members) == 1):
        raise ValueError('Final member count disagrees with result kind')
    seen_nodes, seen_dirs, seen_ranks, weights, predictions, raw_predictions = set(), set(), set(), [], [], []
    for member in members:
        if not isinstance(member, dict):
            raise ValueError('Invalid final member')
        node, directory, rank = member.get('node_id'), member.get('dir'), member.get('rank')
        if not _component(node) or not _component(directory) or type(rank) is not int or rank < 1:
            raise ValueError('Invalid final member identity')
        if directory != f'rank_{rank:02d}_{node}' or node in seen_nodes or directory in seen_dirs or rank in seen_ranks:
            raise ValueError('Duplicate or mismatched final member identity')
        score, weight = member.get('score'), member.get('weight')
        if not _accuracy(score) or not finite_number(weight) or weight <= 0:
            raise ValueError('Invalid final member score or weight')
        prefix = 'final/members/' + directory + '/'
        meta = read_json(inside_file(core, prefix + 'score.json'))
        if not isinstance(meta, dict) or meta.get('node_id') != node or type(meta.get('rank')) is not int or meta['rank'] != rank or not _same_score(meta.get('score'), score):
            raise ValueError('Member score.json disagrees with final source')
        if node == best_node and not _same_score(score, best_score):
            raise ValueError('Final best-member score disagrees with best result')
        if not inside_file(core, prefix + 'solution.py').read_text(encoding='utf-8').strip():
            raise ValueError('Missing final member full solution')
        pred = validate_predictions(inside_file(core, prefix + 'predictions.csv'), sample, target, classes)
        if node == best_node:
            if pred != best_pred:
                raise ValueError('Final best-member predictions differ from the best result')
            if inside_file(core, prefix + 'solution.py').read_bytes() != inside_file(core, 'best/solution.py').read_bytes():
                raise ValueError('Final best-member script differs from the best script')
        seen_nodes.add(node); seen_dirs.add(directory); seen_ranks.add(rank)
        weights.append(weight); predictions.append(pred)
        _, raw_rows = read_csv(inside_file(core, prefix + 'predictions.csv'))
        raw_predictions.append({row['id']: row[target] for row in raw_rows})
    if not math.isfinite(sum(weights)):
        raise ValueError('Nonfinite total final weight')
    if kind == 'single':
        if members[0]['node_id'] != best_node or predictions[0] != best_pred or primary != best_pred:
            raise ValueError('Single final result does not match the best solution')
    else:
        fmt = source.get('format')
        if not isinstance(fmt, dict) or fmt.get('id_col') != 'id' or fmt.get('pred_cols') != [target] or fmt.get('columns') != ['id', target] or fmt.get('copy_cols') != [] or fmt.get('strategy') not in ('vote', 'string_vote', 'prob_argmax') or fmt.get('normalize') is not False:
            raise ValueError('Course class-label ensemble must declare the exact voting format')
        if fmt.get('strategy') == 'prob_argmax':
            prob_cols = fmt.get('prob_cols')
            if not isinstance(prob_cols, list) or len(prob_cols) != len(classes) or len(set(prob_cols)) != len(prob_cols):
                raise ValueError('Invalid prob_argmax class columns')
            if {int(float(c)) for c in prob_cols} != set(classes):
                raise ValueError('prob_argmax columns disagree with course classes')
            member_probs = []
            for member in members:
                p_cols, p_rows = read_csv(inside_file(core, 'final/members/' + member['dir'] + '/test_probabilities.csv'))
                if p_cols != ['id', *prob_cols] or len(p_rows) != len(primary):
                    raise ValueError('Invalid member test_probabilities.csv schema or row count')
                by_id = {}
                for r in p_rows:
                    vec = [float(r[c]) for c in prob_cols]
                    s_vec = sum(vec)
                    if any(not math.isfinite(x) or x < -1e-3 or x > 1.0 + 1e-3 for x in vec) or abs(s_vec - 1.0) > 0.05 or s_vec <= 0:
                        raise ValueError('Invalid probabilities in member test_probabilities.csv')
                    by_id[r['id']] = [x / s_vec for x in vec]
                if set(by_id) != set(primary):
                    raise ValueError('Member test_probabilities.csv IDs disagree with sample')
                member_probs.append(by_id)
            for row_id, actual in primary.items():
                agg = [0.0] * len(prob_cols)
                for m_prob, weight in zip(member_probs, weights):
                    vec = m_prob[row_id]
                    for idx_c, val in enumerate(vec):
                        agg[idx_c] += val * weight
                best_idx = max(range(len(prob_cols)), key=lambda i: agg[i])
                expected = int(float(prob_cols[best_idx]))
                if actual != expected:
                    raise ValueError('Final prediction disagrees with its recorded member probability average')
        else:
            # Verify the saved combination without executing the supplied Python script.
            for row_id, actual in primary.items():
                votes = {}
                for pred, weight in zip(raw_predictions, weights):
                    label = pred[row_id]
                    votes[label] = votes.get(label, 0.) + weight
                # Core votes literal CSV strings, including distinct '0' and '0.0'.
                expected = int(float(max(votes, key=votes.get)))  # first member wins ties
                if actual != expected:
                    raise ValueError('Final prediction disagrees with its recorded member vote')

    # Materialize only documented files, keeping the delivery usable after Core runs/ is removed.
    _plain_tree(core / 'best', destination / 'best')
    dest_final = destination / 'final'
    dest_final.mkdir()
    for name in ('source.json', 'combine.py', 'predictions.csv'):
        shutil.copyfile(inside_file(core, 'final/' + name), dest_final / name)
    artifact_presence = []
    for member in members:
        relative = 'final/members/' + member['dir']
        src = core / relative
        dst = dest_final / 'members' / member['dir']
        dst.mkdir(parents=True)
        for name in ('solution.py', 'score.json', 'predictions.csv', 'val_predictions.csv', 'val_labels.csv', 'val_probabilities.csv', 'test_probabilities.csv'):
            candidate = src / name
            if name in ('solution.py', 'score.json', 'predictions.csv') or candidate.exists() or candidate.is_symlink():
                shutil.copyfile(inside_file(core, relative + '/' + name), dst / name)
        artifact_presence.append(_copy_member_artifacts(core, src, member['node_id'], dst / 'artifacts'))
    shutil.copyfile(dest_final / 'predictions.csv', destination / 'predictions.csv')
    selected = {'source': 'final', 'kind': kind, 'method': source.get('method'), 'metric': source.get('metric'),
                'validation_score': selected_score, 'node_ids': [m['node_id'] for m in members],
                'reason': source.get('reason'), 'members_with_artifacts': sum(artifact_presence),
                'member_count': len(members)}
    return selected, all(artifact_presence), 'full_member_scripts_plus_combine'
