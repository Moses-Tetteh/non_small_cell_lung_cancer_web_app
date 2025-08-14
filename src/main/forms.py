from django import forms
from .models import CTScanRecord

class CTScanForm(forms.ModelForm):
    class Meta:
        model = CTScanRecord
        fields = ['pimage']
        widgets = {
            'pimage': forms.ClearableFileInput(attrs={'accept': 'image/*'})
        }
