from fastapi import APIRouter, HTTPException, status
from typing import Any, Dict, List, Union
import os
import logging

import pandas as pd

router = APIRouter()

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join('core', 'training', 'models', 'xgboost_classifier.pkl')

_MODEL: Any = None
_IS_BOOSTER = False


def load_model(model_path: str = MODEL_PATH):
	"""Try to load model as a joblib/sklearn estimator first. Fall back to XGBoost Booster."""
	global _MODEL, _IS_BOOSTER
	if _MODEL is not None:
		return _MODEL

	# Try joblib (pickle of sklearn wrapper)
	try:
		import joblib
		if os.path.exists(model_path):
			_MODEL = joblib.load(model_path)
			_IS_BOOSTER = False
			logger.info('Loaded model with joblib from %s', model_path)
			return _MODEL
	except Exception:
		logger.debug('joblib load failed or joblib not available')

	# Try XGBoost native load -> create XGBClassifier and call load_model
	try:
		import xgboost as xgb
		if os.path.exists(model_path):
			clf = xgb.XGBClassifier()
			clf.load_model(model_path)
			_MODEL = clf
			_IS_BOOSTER = False
			logger.info('Loaded model as xgboost.XGBClassifier from %s', model_path)
			return _MODEL
	except Exception:
		logger.debug('xgboost sklearn-wrapper load failed or xgboost not installed')

	# Finally try Booster format
	try:
		import xgboost as xgb
		if os.path.exists(model_path):
			booster = xgb.Booster()
			booster.load_model(model_path)
			_MODEL = booster
			_IS_BOOSTER = True
			logger.info('Loaded model as xgboost.Booster from %s', model_path)
			return _MODEL
	except Exception:
		logger.debug('xgboost Booster load failed or xgboost not installed')

	# If nothing succeeded, leave _MODEL as None
	logger.warning('No usable model found at %s', model_path)
	return None


def _predict_from_model(X: pd.DataFrame) -> Dict[str, List[Union[int, float]]]:
	"""Return predictions and probabilities for a DataFrame X.

	Returns dict with keys: 'preds' (list of ints) and 'probs' (list of floats, probability for positive class when available).
	"""
	model = load_model()
	if model is None:
		raise RuntimeError('Model is not available. Ensure a trained model exists at the configured path.')

	if not _IS_BOOSTER:
		# sklearn-like API
		if hasattr(model, 'predict_proba'):
			probs = model.predict_proba(X)[:, 1].tolist()
		else:
			# fallback to decision_function
			if hasattr(model, 'decision_function'):
				scores = model.decision_function(X)
				scores = (scores - scores.min()) / (scores.max() - scores.min())
				probs = scores.tolist()
			else:
				probs = [None] * len(X)
		preds = model.predict(X).tolist()
	else:
		# xgboost.Booster: use DMatrix
		import xgboost as xgb
		dmat = xgb.DMatrix(X)
		probs = model.predict(dmat).tolist()
		preds = [int(p >= 0.5) for p in probs]

	return {'preds': preds, 'probs': probs}



@router.get('/year/{year}/month/{month}/depth/{depth}')
def get_classifier_data(year: int, month: int, depth: int):
	"""Simple GET endpoint that builds a minimal feature dict from the path params and returns prediction.

	Note: the model expects a specific set/order of features. This endpoint creates a minimal feature set
	(e.g. year, month, depth) which will only work if the model was trained with these exact column names.
	For more flexible usage, use the POST /predict endpoint below and provide a features JSON object.
	"""
	# Build a single-row DataFrame from provided path params
	features = {'year': year, 'month': month, 'depth': depth}
	X = pd.DataFrame([features])

	try:
		out = _predict_from_model(X)
	except Exception as e:
		logger.exception('Prediction failed')
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

	return {'input': features, 'prediction': out['preds'][0], 'probability': out['probs'][0]}


@router.post('/predict')
def predict(features: Union[Dict[str, Any], List[Dict[str, Any]]]):
	"""POST endpoint: accepts a single feature dict or a list of feature dicts and returns predictions.

	Example payloads:
	  {"lat": 10.3, "lon": -45.2, "depth": 30, "hour": 14}
	or
	  [{...}, {...}, ...]
	"""
	# Normalize to list of dicts
	if isinstance(features, dict):
		feats = [features]
	elif isinstance(features, list):
		feats = features
	else:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid payload format')

	X = pd.DataFrame(feats)

	try:
		out = _predict_from_model(X)
	except RuntimeError as e:
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
	except Exception as e:
		logger.exception('Prediction failed')
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

	return {'predictions': out['preds'], 'probabilities': out['probs']}
