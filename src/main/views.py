import logging
from django.shortcuts import render
from django.http import JsonResponse
from .forms import CTScanForm

logger = logging.getLogger(__name__)

def home_page(request):
    if request.method == 'POST':
        form = CTScanForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                scan_record = form.save()  # Your model saves and triggers classification
                
                uploaded_image = scan_record.pimage.url if scan_record.pimage else None
                prediction = scan_record.classified
                confidence = getattr(scan_record, 'confidence', None)  # Fetch confidence
                
                # Fallback if confidence missing
                confidence_display = f"{confidence:.2%}" if confidence is not None else "N/A"

                # Recommendation logic
                recommendations = {
                    'adenocarcinoma': (
                        "Your scan indicates adenocarcinoma, a common subtype of non-small cell lung cancer. "
                        "We recommend consulting an oncologist to discuss targeted therapies, which may include "
                        "molecular testing to identify specific mutations that can guide personalized treatment. "
                        "Early intervention can improve outcomes."
                    ),
                    'large_cell_carcinoma': (
                        "The scan suggests large cell carcinoma, a fast-growing type of lung cancer. "
                        "Additional diagnostic tests such as a biopsy are highly recommended to confirm the diagnosis "
                        "and determine the cancer's aggressiveness. Your healthcare provider will guide you on the best "
                        "treatment approach."
                    ),
                    'normal': (
                        "No signs of lung cancer were detected in the CT scan. Routine monitoring with periodic imaging "
                        "is advised to ensure continued lung health. Maintain a healthy lifestyle and consult your doctor "
                        "if you experience any symptoms or changes."
                    ),
                    'squamous_cell_carcinoma': (
                        "The prediction indicates squamous cell carcinoma, another subtype of non-small cell lung cancer. "
                        "Treatment often involves chemotherapy, radiation, or a combination, depending on the stage and "
                        "location of the tumor. Please consult with your oncologist for a personalized treatment plan."
                    ),
                }

                recommendation = recommendations.get(prediction, "Please consult your doctor.")

                # AJAX JSON response
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({
                        'image_url': uploaded_image,
                        'prediction': prediction,
                        'confidence': confidence_display,
                        'recommendation': recommendation,
                    })

                # Non-AJAX fallback
                context = {
                    'form': form,
                    'uploaded_image': uploaded_image,
                    'prediction': prediction,
                    'confidence': confidence_display,
                    'recommendation': recommendation,
                    'error_message': None,
                }
                return render(request, 'main/home.html', context)

            except Exception as e:
                logger.error(f"Error processing scan: {e}", exc_info=True)
                error_message = "Error processing scan. Please try again."
        else:
            error_message = "Invalid form submission. Please upload a valid image."

        # Handle errors
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'error': error_message}, status=400)

        context = {
            'form': form,
            'uploaded_image': None,
            'prediction': None,
            'confidence': None,
            'recommendation': None,
            'error_message': error_message,
        }
        return render(request, 'main/home.html', context)

    # GET request
    form = CTScanForm()
    return render(request, 'main/home.html', {'form': form})
