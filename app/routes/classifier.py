from fastapi import APIRouter, HTTPException, status
from typing import Any, Dict, List, Union
import os
import logging

import pandas as pd

from app.models import ClassifierDataRequest, ClassifierDataResponse, SpotData
import datetime as dt

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


@router.post('/classify', response_model=ClassifierDataResponse)
def classify(req: ClassifierDataRequest):
	"""Classify a list of coordinates using the trained model and return SpotData list.

	The request contains `coordinates` (list of [longitude, latitude]), a `view` string, a `date` and `depth`.
	For each coordinate we build a minimal feature row with longitude, latitude, depth, and date-derived fields
	(year, month) to feed the model. The returned `count` in `SpotData` is the predicted probability scaled to 0-100
	(int) when probability is available, otherwise the binary prediction (0/1).
	"""
	# Normalize coordinates; expect [longitude, latitude]
	coords = req.coords
	if not coords:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='No coordinates provided')

	# We'll batch call the meteomatics API for all coordinates at once for the requested date
	# meteomatics_get_data expects coordinates as [(lat, lon), ...] and start/end datetimes
	try:
		# prepare coordinates as list of (lat, lon)
		mm_coords = [(float(c[1]), float(c[0])) for c in coords]
	except Exception:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Each coordinate must be [longitude, latitude]')

	# Use the provided date (datetime) as start and end
	if isinstance(req.date, dt.datetime):
		start_date = req.date
		end_date = req.date
	else:
		try:
			start_date = dt.datetime.fromisoformat(req.date)
			end_date = start_date
		except Exception:
			start_date = dt.datetime.now()
			end_date = start_date

	# Import meteomatics helper lazily so missing optional deps don't break module import
	try:
		from core.data.get_meteomatics import get_data as meteomatics_get_data
	except Exception:
		meteomatics_get_data = None

	if meteomatics_get_data is not None:
		try:
			mm_results = meteomatics_get_data(mm_coords, start_date, end_date)
   			print('mm_results', mm_results)
		except Exception:
			logger.exception('Failed to call meteomatics API')
			mm_results = []
	else:
		logger.info('meteomatics_get_data not available; skipping external data fetch')
		mm_results = []

	# Map meteomatics results back to provided coordinates. meteomatics returns multiple entries (one per coord per date)
	# We'll choose the first matching entry per coordinate
	coord_to_entry = {}
	for entry in mm_results:
		key = (round(entry['lon'], 5), round(entry['lat'], 5))
		if key not in coord_to_entry:
			coord_to_entry[key] = entry

	rows = []
	for c in coords:
		longitude, latitude = float(c[0]), float(c[1])
		key = (round(longitude, 5), round(latitude, 5))
		entry = coord_to_entry.get(key)
		if entry:
			row = {
				'longitude': longitude,
				'latitude': latitude,
				'depth': req.depth,
				'temperature': entry.get('temperature', 0.0),
				'max_individual_wave_height': entry.get('max_individual_wave_height', 0.0),
				'mean_wave_direction': entry.get('mean_wave_direction', 0.0),
				'mean_period_total_swell': entry.get('mean_period_total_swell', 0.0),
				'clouds': entry.get('clouds', 0.0),
				'year': start_date.year,
				'month': start_date.month,
			}
		else:
			# fallback minimal row
			row = {
				'longitude': longitude,
				'latitude': latitude,
				'depth': req.depth,
				'temperature': 0.0,
				'max_individual_wave_height': 0.0,
				'mean_wave_direction': 0.0,
				'mean_period_total_swell': 0.0,
				'clouds': 0.0,
				'year': start_date.year,
				'month': start_date.month,
			}
		rows.append(row)

	X = pd.DataFrame(rows)

	try:
		out = _predict_from_model(X)
	except Exception as e:
		logger.exception('Classification failed')
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

	preds = out.get('preds', [])
	probs = out.get('probs', [])

	spots: List[SpotData] = []
	for i, row in enumerate(rows):
		pred = preds[i] if i < len(preds) else 0
		prob = None
		if i < len(probs) and probs[i] is not None:
			try:
				prob = float(probs[i])
			except Exception:
				prob = None

		if prob is not None:
			count = int(round(prob * 100))
		else:
			count = int(pred)

		spots.append(SpotData(longitude=row['longitude'], latitude=row['latitude'], count=count))

	return ClassifierDataResponse(data=spots)
