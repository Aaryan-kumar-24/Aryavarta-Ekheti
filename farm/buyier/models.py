from django.db import models
from django.contrib.auth.models import User
class Buyier(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # buyer user
    buyer_name = models.CharField(max_length=100)
    buyer_address = models.CharField(max_length=100)
    purchase_quantity = models.PositiveIntegerField()
    negotiation_price = models.PositiveIntegerField()
    farmer_name = models.CharField(max_length=100)
    crop_name = models.CharField(max_length=100)
    buyer_phone = models.PositiveIntegerField()

    status = models.CharField(max_length=20, default="pending")  # ✅ NEW FIELD

    def __str__(self):
        return f"{self.buyer_name} -> {self.crop_name}"

