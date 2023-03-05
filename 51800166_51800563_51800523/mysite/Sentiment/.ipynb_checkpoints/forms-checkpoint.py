from django import forms
 
class RegisterForm(forms.Form):
    link = forms.CharField(widget = forms.TextInput(attrs={'class':'custom-form'}), label = 'Link youtube', max_length = 300)