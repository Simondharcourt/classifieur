import React, { useState } from 'react';
import { validateClassifications } from '../api/api';
import { ValidationRequest, ValidationResponse } from '../types/api';

const Validate: React.FC = () => {
  const [text, setText] = useState('');
  const [categories, setCategories] = useState<string[]>([]);
  const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleValidate = async () => {
    if (!text) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const texts = text.split('\n').filter(t => t.trim());
      const samples = texts.map(t => ({
        text: t,
        assigned_category: '', // This would be filled with actual classifications
        confidence: 0
      }));

      const request: ValidationRequest = {
        samples,
        current_categories: categories,
        text_columns: ['text']
      };

      const response = await validateClassifications(request);
      setValidationResult(response);
    } catch (err) {
      setError('Failed to validate classifications');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900">Validate Classifications</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>Enter text samples (one per line) to validate their classifications.</p>
          </div>
          <div className="mt-5">
            <textarea
              rows={4}
              className="shadow-sm focus:ring-primary-500 focus:border-primary-500 block w-full sm:text-sm border-gray-300 rounded-md"
              placeholder="Enter text samples..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
          <div className="mt-5">
            <button
              type="button"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              onClick={handleValidate}
              disabled={loading}
            >
              Validate
            </button>
          </div>
        </div>
      </div>

      {loading && (
        <div className="flex justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {validationResult && (
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Validation Results</h3>
          </div>
          <div className="border-t border-gray-200">
            <dl>
              {validationResult.accuracy_score !== undefined && (
                <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                  <dt className="text-sm font-medium text-gray-500">Accuracy Score</dt>
                  <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                    {(validationResult.accuracy_score * 100).toFixed(2)}%
                  </dd>
                </div>
              )}
              <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Validation Report</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  {validationResult.validation_report}
                </dd>
              </div>
              {validationResult.misclassifications && validationResult.misclassifications.length > 0 && (
                <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                  <dt className="text-sm font-medium text-gray-500">Misclassifications</dt>
                  <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                    <ul className="border border-gray-200 rounded-md divide-y divide-gray-200">
                      {validationResult.misclassifications.map((item, index) => (
                        <li key={index} className="pl-3 pr-4 py-3 flex items-center justify-between text-sm">
                          <div className="w-0 flex-1 flex items-center">
                            <span className="ml-2 flex-1 w-0 truncate">
                              {item.text}
                            </span>
                          </div>
                          <div className="ml-4 flex-shrink-0">
                            <span className="text-gray-500">
                              Current: {item.current_category}
                            </span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </dd>
                </div>
              )}
              {validationResult.suggested_improvements && validationResult.suggested_improvements.length > 0 && (
                <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                  <dt className="text-sm font-medium text-gray-500">Suggested Improvements</dt>
                  <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                    <ul className="list-disc pl-5 space-y-1">
                      {validationResult.suggested_improvements.map((improvement, index) => (
                        <li key={index}>{improvement}</li>
                      ))}
                    </ul>
                  </dd>
                </div>
              )}
            </dl>
          </div>
        </div>
      )}
    </div>
  );
};

export default Validate; 