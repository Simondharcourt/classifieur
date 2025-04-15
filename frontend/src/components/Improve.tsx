import React, { useState } from 'react';
import { improveClassification } from '../api/api';
import { ImprovementRequest, ImprovementResponse } from '../types/api';

const Improve: React.FC = () => {
  const [text, setText] = useState('');
  const [categories, setCategories] = useState('');
  const [validationReport, setValidationReport] = useState('');
  const [improvementResult, setImprovementResult] = useState<ImprovementResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImprove = async () => {
    if (!text || !categories || !validationReport) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const request: ImprovementRequest = {
        df: { text: text.split('\n').filter(t => t.trim()) },
        validation_report: validationReport,
        text_columns: ['text'],
        categories,
        classifier_type: 'gpt35',
        show_explanations: true,
        file_path: 'examples/emails.csv'
      };

      const response = await improveClassification(request);
      setImprovementResult(response);
    } catch (err) {
      setError('Failed to improve classifications');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900">Improve Classifications</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>Enter text samples and validation report to improve classifications.</p>
          </div>
          <div className="mt-5 space-y-4">
            <div>
              <label htmlFor="text" className="block text-sm font-medium text-gray-700">
                Text Samples
              </label>
              <div className="mt-1">
                <textarea
                  id="text"
                  rows={4}
                  className="shadow-sm focus:ring-primary-500 focus:border-primary-500 block w-full sm:text-sm border-gray-300 rounded-md"
                  placeholder="Enter text samples (one per line)..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
              </div>
            </div>
            <div>
              <label htmlFor="categories" className="block text-sm font-medium text-gray-700">
                Categories
              </label>
              <div className="mt-1">
                <input
                  type="text"
                  id="categories"
                  className="shadow-sm focus:ring-primary-500 focus:border-primary-500 block w-full sm:text-sm border-gray-300 rounded-md"
                  placeholder="Enter categories (comma-separated)"
                  value={categories}
                  onChange={(e) => setCategories(e.target.value)}
                />
              </div>
            </div>
            <div>
              <label htmlFor="validation-report" className="block text-sm font-medium text-gray-700">
                Validation Report
              </label>
              <div className="mt-1">
                <textarea
                  id="validation-report"
                  rows={4}
                  className="shadow-sm focus:ring-primary-500 focus:border-primary-500 block w-full sm:text-sm border-gray-300 rounded-md"
                  placeholder="Enter validation report..."
                  value={validationReport}
                  onChange={(e) => setValidationReport(e.target.value)}
                />
              </div>
            </div>
          </div>
          <div className="mt-5">
            <button
              type="button"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              onClick={handleImprove}
              disabled={loading}
            >
              Improve
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

      {improvementResult && (
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Improvement Results</h3>
          </div>
          <div className="border-t border-gray-200">
            <dl>
              <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Success</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  {improvementResult.success ? 'Yes' : 'No'}
                </dd>
              </div>
              <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">New Validation Report</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  {improvementResult.new_validation_report}
                </dd>
              </div>
              {improvementResult.updated_categories.length > 0 && (
                <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                  <dt className="text-sm font-medium text-gray-500">Updated Categories</dt>
                  <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                    <div className="flex flex-wrap gap-2">
                      {improvementResult.updated_categories.map((category, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium bg-primary-100 text-primary-800"
                        >
                          {category}
                        </span>
                      ))}
                    </div>
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

export default Improve; 