from flask import Blueprint, request, jsonify, send_from_directory, current_app
import os
import uuid
import tempfile
import shutil
import json
import logging
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename
from app.config import Config
from app.modules.data_processing import process_review_paper, process_multiple_papers, normalize_entities, transform_table_data_to_entities, save_data_to_json
from app.modules.knowledge_graph import build_knowledge_graph, visualize_graph, export_graph_to_json
from app.modules.db_manager import db_manager
from app.modules.agents import extract_paper_entities, extract_text_from_pdf

# 创建蓝图
combined_api = Blueprint('combined_api', __name__) 